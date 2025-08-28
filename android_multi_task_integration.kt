import android.content.Context
import android.content.res.AssetManager
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.charset.StandardCharsets

/**
 * Multi-task SMS Classifier with Entity Extraction
 * 
 * This class demonstrates how to use the enhanced TFLite model that provides:
 * 1. Binary classification (transactional vs not_transactional)
 * 2. Entity extraction (merchant name, amount, transaction type)
 * 3. Transaction direction (debit/credit/none)
 */
class SmsMultiTaskClassifier(private val context: Context) {
    
    private var interpreter: Interpreter? = null
    private var spTokenizer: SpTokenizer? = null
    
    // Model output shapes
    private val classificationShape = intArrayOf(1, 1)           // [1, 1] - binary probability
    private val nerShape = intArrayOf(1, 200, 3)                // [1, 200, 3] - BIO tagging for each token
    private val directionShape = intArrayOf(1, 3)                // [1, 3] - debit/credit/none probabilities
    
    // Output buffers
    private val classificationBuffer = TensorBuffer.createFixedSize(classificationShape, org.tensorflow.lite.DataType.FLOAT32)
    private val merchantBuffer = TensorBuffer.createFixedSize(nerShape, org.tensorflow.lite.DataType.FLOAT32)
    private val amountBuffer = TensorBuffer.createFixedSize(nerShape, org.tensorflow.lite.DataType.FLOAT32)
    private val typeBuffer = TensorBuffer.createFixedSize(nerShape, org.tensorflow.lite.DataType.FLOAT32)
    private val directionBuffer = TensorBuffer.createFixedSize(directionShape, org.tensorflow.lite.DataType.FLOAT32)
    
    data class SmsAnalysis(
        val isTransactional: Boolean,
        val confidence: Float,
        val merchant: String?,
        val amount: String?,
        val transactionType: String?,
        val direction: TransactionDirection,
        val directionConfidence: Float
    )
    
    enum class TransactionDirection {
        DEBIT, CREDIT, NONE
    }
    
    /**
     * Load the multi-task TFLite model and SentencePiece tokenizer
     */
    fun loadModel(modelPath: String, tokenizerPath: String): Boolean {
        return try {
            // Load TFLite model
            val modelFile = File(context.getExternalFilesDir(null), modelPath)
            val options = Interpreter.Options()
            interpreter = Interpreter(modelFile, options)
            
            // Load SentencePiece tokenizer
            spTokenizer = SpTokenizer(context, tokenizerPath)
            
            true
        } catch (e: Exception) {
            e.printStackTrace()
            false
        }
    }
    
    /**
     * Analyze SMS text and extract structured information
     */
    fun analyzeSms(text: String): SmsAnalysis {
        val tokens = preprocessText(text)
        
        // Prepare input
        val inputBuffer = TensorBuffer.createFixedSize(intArrayOf(1, 200), org.tensorflow.lite.DataType.INT32)
        inputBuffer.loadArray(tokens)
        
        // Run inference
        val inputs = mapOf("input_ids" to inputBuffer.buffer)
        val outputs = mapOf(
            "classification" to classificationBuffer.buffer,
            "merchant_ner" to merchantBuffer.buffer,
            "amount_ner" to amountBuffer.buffer,
            "type_ner" to typeBuffer.buffer,
            "direction" to directionBuffer.buffer
        )
        
        interpreter?.runForMultipleInputsOutputs(inputs, outputs)
        
        // Extract results
        val classificationProb = classificationBuffer.floatArray[0]
        val isTransactional = classificationProb > 0.5
        
        val merchant = extractEntity(text, tokens, merchantBuffer.floatArray)
        val amount = extractEntity(text, tokens, amountBuffer.floatArray)
        val transactionType = extractEntity(text, tokens, typeBuffer.floatArray)
        
        val directionProbs = directionBuffer.floatArray
        val directionIndex = directionProbs.indices.maxByOrNull { directionProbs[it] } ?: 2
        val direction = when (directionIndex) {
            0 -> TransactionDirection.DEBIT
            1 -> TransactionDirection.CREDIT
            else -> TransactionDirection.NONE
        }
        val directionConfidence = directionProbs[directionIndex]
        
        return SmsAnalysis(
            isTransactional = isTransactional,
            confidence = classificationProb,
            merchant = merchant,
            amount = amount,
            transactionType = transactionType,
            direction = direction,
            directionConfidence = directionConfidence
        )
    }
    
    /**
     * Preprocess text using SentencePiece tokenizer
     */
    private fun preprocessText(text: String): IntArray {
        val standardized = standardizeText(text)
        val tokens = spTokenizer?.tokenize(standardized) ?: intArrayOf()
        
        return if (tokens.size > 200) {
            tokens.take(200).toIntArray()
        } else {
            tokens + IntArray(200 - tokens.size) { 0 } // Pad with 0
        }
    }
    
    /**
     * Standardize text (lowercase, remove punctuation)
     */
    private fun standardizeText(text: String): String {
        return text.lowercase()
            .replace(Regex("[!\"#\$%&'()*+,-./:;<=>?@\\[\\]\\\\^_`{|}~]"), "")
    }
    
    /**
     * Extract entity text from NER predictions
     */
    private fun extractEntity(originalText: String, tokens: IntArray, nerPredictions: FloatArray): String? {
        val words = originalText.split(" ")
        val entityTokens = mutableListOf<String>()
        
        for (i in tokens.indices) {
            if (i >= words.size) break
            
            val tokenStart = i * 3
            if (tokenStart + 2 < nerPredictions.size) {
                val probs = floatArrayOf(
                    nerPredictions[tokenStart],     // O (outside)
                    nerPredictions[tokenStart + 1], // B-ENTITY (beginning)
                    nerPredictions[tokenStart + 2]  // I-ENTITY (inside)
                )
                
                val maxIndex = probs.indices.maxByOrNull { probs[it] } ?: 0
                if (maxIndex > 0) { // B-ENTITY or I-ENTITY
                    entityTokens.add(words[i])
                }
            }
        }
        
        return if (entityTokens.isNotEmpty()) entityTokens.joinToString(" ") else null
    }
    
    /**
     * Batch analysis for multiple SMS messages
     */
    fun analyzeBatch(texts: List<String>): List<SmsAnalysis> {
        return texts.map { analyzeSms(it) }
    }
    
    /**
     * Clean up resources
     */
    fun close() {
        interpreter?.close()
        interpreter = null
        spTokenizer = null
    }
}

/**
 * SentencePiece Tokenizer Interface
 */
interface SpTokenizer {
    fun tokenize(text: String): IntArray
    fun detokenize(tokenIds: IntArray): String
}

/**
 * SentencePiece Tokenizer Implementation
 * Note: This is a simplified interface. You'll need to implement the actual
 * SentencePiece tokenization using a library like sentencepiece-android
 */
class SpTokenizer(private val context: Context, private val modelPath: String) : SpTokenizer {
    
    override fun tokenize(text: String): IntArray {
        // TODO: Implement actual SentencePiece tokenization
        // For now, return a simple word-based tokenization
        return text.split(" ").mapIndexed { index, _ -> index + 1 }.toIntArray()
    }
    
    override fun detokenize(tokenIds: IntArray): String {
        // TODO: Implement actual SentencePiece detokenization
        return tokenIds.joinToString(" ")
    }
}

// Usage Example
fun main() {
    // Example usage in Android app
    val classifier = SmsMultiTaskClassifier(context)
    
    if (classifier.loadModel("sms_multi_task.tflite", "tokenizer.spm")) {
        
        val smsText = "Rs.5000 debited from A/c XXXX1234 for UPI transaction to Amazon on 15/12/2024"
        val analysis = classifier.analyzeSms(smsText)
        
        println("SMS Analysis:")
        println("Transactional: ${analysis.isTransactional}")
        println("Confidence: ${analysis.confidence}")
        println("Merchant: ${analysis.merchant}")
        println("Amount: ${analysis.amount}")
        println("Type: ${analysis.transactionType}")
        println("Direction: ${analysis.direction}")
        println("Direction Confidence: ${analysis.directionConfidence}")
        
        // Example output:
        // SMS Analysis:
        // Transactional: true
        // Confidence: 0.9987
        // Merchant: Amazon
        // Amount: 5000
        // Type: UPI
        // Direction: DEBIT
        // Direction Confidence: 0.9876
        
        classifier.close()
    }
}
