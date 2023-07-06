using System;
using System.Collections.Generic;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using NumSharp;
using F23.StringSimilarity;

namespace Samples.Dynamic
{
    public static class ApplyWordEmbedding
    {
        public static void Main(string[] args)
        {
            Example();
        }

        public static void Example()
        {
            var mlContext = new MLContext();

            // Create an empty list as the dataset for TextData. The 'ApplyWordEmbedding' does
            // not require training data as the estimator ('WordEmbeddingEstimator')
            // created by 'ApplyWordEmbedding' API is not a trainable estimator.
            // The empty list is only needed to pass input schema to the pipeline.
            var emptyTextDataSamples = new List<TextData>();

            // Convert sample list to an empty IDataView.
            var emptyTextDataView = mlContext.Data.LoadFromEnumerable(emptyTextDataSamples);

            // A pipeline for converting text into a 150-dimension embedding vector
            // using pretrained 'GloVeTwitter25D' model. The
            // 'ApplyWordEmbedding' computes the minimum, average and maximum values
            // for each token's embedding vector. Tokens in 
            // 'GloVeTwitter25D' model are represented as
            // 25-dimension vector. Therefore, the output is of 75-dimension [min,
            // avg, max].
            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.FastTextWikipedia300D));

            // Fit to data.
            var textTransformer = textPipeline.Fit(emptyTextDataView);

            // Create the prediction engine for TextData to get the embedding vector from the input text/string.
            var textPredictionEngine = mlContext.Model.CreatePredictionEngine<TextData, TransformedTextData>(textTransformer);

            // Call the prediction API to convert the text into embedding vector.
            var data = new TextData()
            {
                Text = "This is a great product. I would like to buy it again."
            };
            var comparingData = new TextData()
            {
                Text = "That sucks. I would like to buy it again."
            };

            var prediction = textPredictionEngine.Predict(data);
            var prediction2 = textPredictionEngine.Predict(comparingData);

            // Print the length of the embedding vector for TextData.
            Console.WriteLine($"Number of Features (TextData): {prediction.Features.Length}");

            // Print the embedding vector for TextData.
            Console.Write("Features (TextData): ");
            foreach (var feature in prediction.Features)
                Console.Write($"{feature:F4} ");
            Console.WriteLine();

            // Print the length of the embedding vector for DataToCompare.
            Console.WriteLine($"Number of Features (DataToCompare): {prediction2.Features.Length}");

            // Print the embedding vector for DataToCompare.
            Console.Write("Features (DataToCompare): ");
            foreach (var feature in prediction2.Features)
                Console.Write($"{feature:F4} ");
            Console.WriteLine();

            // Compute the cosine similarity between the two embedding vectors.
            var cosineSimilarity = CalculateCosineSimilarity(prediction.Features, prediction2.Features);

            // Compute the cosine similarity between the two embedding vectors using NumSharp.
            var cosineSimilarityNumSharp = CalculateCosineSimilarityNumSharp(prediction.Features, prediction2.Features);

            // Compute the Levenshtein similarity between the two texts.
            var levenshteinSimilarity = CalculateLevenshteinSimilarity(data.Text, comparingData.Text);

            // Print the cosine similarity.
            Console.WriteLine($"Cosine Similarity: {cosineSimilarity:F4}");

            // Print the cosine similarity using NumSharp.
            Console.WriteLine($"Cosine Similarity (NumSharp): {cosineSimilarityNumSharp:F4}");

            // Print the Levenshtein similarity.
            Console.WriteLine($"Levenshtein Similarity: {levenshteinSimilarity:F4}");
        }

        private static float CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1.Length != vector2.Length)
                throw new ArgumentException("The vectors must have the same length.");

            var dotProduct = 0.0f;
            var magnitude1 = 0.0f;
            var magnitude2 = 0.0f;

            for (int i = 0; i < vector1.Length; i++)
            {
                dotProduct += vector1[i] * vector2[i];
                magnitude1 += vector1[i] * vector1[i];
                magnitude2 += vector2[i] * vector2[i];
            }

            magnitude1 = (float)Math.Sqrt(magnitude1);
            magnitude2 = (float)Math.Sqrt(magnitude2);

            return dotProduct / (magnitude1 * magnitude2);
        }

        private static double CalculateCosineSimilarityNumSharp(float[] vector1, float[] vector2)
        {
            var npVector1 = np.array(vector1);
            var npVector2 = np.array(vector2);

            double magnitude1 = np.sqrt(np.sum(np.power(npVector1, 2)));
            double magnitude2 = np.sqrt(np.sum(np.power(npVector2, 2)));

            double dotProduct = np.sum(npVector1 * npVector2);
            double cosineSimilarity = dotProduct / (magnitude1 * magnitude2);

            return cosineSimilarity;
        }

        private static double CalculateLevenshteinSimilarity(string text1, string text2)
        {
            var normalizedLevenshtein = new NormalizedLevenshtein();
            var similarity = normalizedLevenshtein.Similarity(text1, text2);
            return similarity;
        }

        private class TextData
        {
            public string Text { get; set; }
        }

        private class TransformedTextData : TextData
        {
            public float[] Features { get; set; }
        }
    }
}
