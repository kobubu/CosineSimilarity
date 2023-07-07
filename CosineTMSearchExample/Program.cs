using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using NPOI.SS.UserModel;
using NPOI.XSSF.UserModel;
using NumSharp;

namespace Samples.Dynamic
{
    public static class ApplyWordEmbedding
    {


        public static void Main(string[] args)
        {
            var stopwatch = new Stopwatch();
            var filePath = "C:\\Users\\Igor\\Desktop\\models\\наборы данных\\En_Es_Deepl_Kefir_Grim_Soul\\Kefir — Grim Soul_translation_memory_en_and_es_translation_and_es_deepl_only_text.xlsx";

            stopwatch.Start();
            var comparingDataList = FindSimilarStringsInExcel(filePath);
            CalculateSimilarities(comparingDataList);
            stopwatch.Stop();

            Console.WriteLine($"Время выполнения операции: {stopwatch.ElapsedMilliseconds} мс");
            Console.WriteLine("finished");
        }

        private static List<ComparingData> FindSimilarStringsInExcel(string filePath)
        {
            var comparingDataList = new List<ComparingData>();

            using (var file = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                var workbook = new XSSFWorkbook(file);
                var sheet = workbook.GetSheetAt(0);

                for (int rowIdx = 1; rowIdx <= sheet.LastRowNum; rowIdx++)
                {
                    var row = sheet.GetRow(rowIdx);
                    if (row != null)
                    {
                        var text = row.GetCell(0)?.StringCellValue;
                        if (!string.IsNullOrEmpty(text))
                        {
                            var comparingData = new ComparingData
                            {
                                Text = text,
                                RowNumber = rowIdx
                            };
                            comparingDataList.Add(comparingData);
                        }
                    }
                }
            }

            return comparingDataList;
        }

        private static void CalculateSimilarities(List<ComparingData> comparingDataList)
        {
            var mlContext = new MLContext();
            var emptyComparingDataSamples = new List<ComparingData>();
            var emptyComparingDataView = mlContext.Data.LoadFromEnumerable(emptyComparingDataSamples);

            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D));

            var textTransformer = textPipeline.Fit(emptyComparingDataView);

            var textPredictionEngine = mlContext.Model.CreatePredictionEngine<ComparingData, TransformedTextData>(textTransformer);

            foreach (var comparingData in comparingDataList)
            {
                var prediction = textPredictionEngine.Predict(comparingData);
                var cosineSimilarity = CalculateCosineSimilarity(prediction.FeaturesA, prediction.FeaturesB);

                comparingData.BestFiveMatches = GetTopFiveSimilarStrings(comparingDataList, comparingData, mlContext, textPredictionEngine);
            }
        }

        private static List<ComparingData> GetTopFiveSimilarStrings(List<ComparingData> comparingDataList, ComparingData targetString, MLContext mlContext, PredictionEngine<ComparingData, TransformedTextData> predictionEngine)
        {
            var cosineSimilarities = new List<float>();

            foreach (var comparingData in comparingDataList)
            {
                if (comparingData != targetString)
                {
                    var prediction = predictionEngine.Predict(comparingData);
                    var cosineSimilarity = CalculateCosineSimilarity(prediction.FeaturesA, prediction.FeaturesB);
                    cosineSimilarities.Add(cosineSimilarity);
                }
            }

            var topFiveSimilarStrings = cosineSimilarities
                .Select((similarity, index) => new { Similarity = similarity, Index = index })
                .OrderByDescending(item => item.Similarity)
                .Take(5)
                .Select(item => comparingDataList[item.Index])
                .ToList();

            return topFiveSimilarStrings;
        }


        public static void CheckPairs(string filePath)
        {


            var mlContext = new MLContext();

            var emptyTextDataSamples = new List<TextDataPair>();

            var emptyTextDataView = mlContext.Data.LoadFromEnumerable(emptyTextDataSamples);

            var textPipeline = mlContext.Transforms.Text.NormalizeText("TextA")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("TokensA", "TextA"))
                .Append(mlContext.Transforms.Text.NormalizeText("TextB"))
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("TokensB", "TextB"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("FeaturesA", "TokensA", WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("FeaturesB", "TokensB", WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D));

            var textTransformer = textPipeline.Fit(emptyTextDataView);

            var textPredictionEngine = mlContext.Model.CreatePredictionEngine<TextDataPair, TransformedTextData>(textTransformer);

            var textDataPairs = LoadDataFromExcel(filePath);

            foreach (var textDataPair in textDataPairs)
            {
                var prediction = textPredictionEngine.Predict(textDataPair);
                var cosineSimilarity = CalculateCosineSimilarity(prediction.FeaturesA, prediction.FeaturesB);


                //Console.WriteLine($"Text A: {textDataPair.TextA}");
                //Console.WriteLine($"Number of Features (TextData): {prediction.FeaturesA.Length}");
                //Console.WriteLine("Text A features: ");
                //foreach (var feature in prediction.FeaturesA)
                //    Console.Write($"{feature:F4} ");
                //Console.WriteLine();
                //Console.WriteLine();
                //Console.WriteLine($"Text B: {textDataPair.TextB}");
                //Console.WriteLine($"Number of Features (TextData): {prediction.FeaturesB.Length}");
                //Console.WriteLine("Text B features: ");
                //foreach (var feature in prediction.FeaturesB)
                //    Console.Write($"{feature:F4} ");
                //Console.WriteLine();
                //Console.WriteLine();
                //Console.WriteLine($"Cosine Similarity between Text B and Text B: {cosineSimilarity:F4}");
                //Console.WriteLine();
            }

            Console.WriteLine("finished");
        }

        private static List<TextDataPair> LoadDataFromExcel(string filePath)
        {
            var textDataPairs = new List<TextDataPair>();

            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                var workbook = new XSSFWorkbook(fileStream);
                var sheet = workbook.GetSheetAt(0);

                for (int rowIdx = 1; rowIdx <= sheet.LastRowNum; rowIdx++)
                {
                    var row = sheet.GetRow(rowIdx);
                    if (row == null)
                        continue;

                    var textA = row.GetCell(0)?.StringCellValue;
                    var textB = row.GetCell(1)?.StringCellValue;
                    var textDataPair = new TextDataPair { TextA = textA, TextB = textB };
                    textDataPairs.Add(textDataPair);
                }
            }
            return textDataPairs;
        }

        private class ComparingData
        {
            public string Text { get; set; }
            public float[] Features { get; set; }
            public float Similarity { get; set; }
            public ComparingData[] BestFiveMatches { get; set; }
            public int RowNumber { get; set; }
        }
        private class TextDataPair : TransformedTextData
        {
            public string TextA { get; set; }
            public string TextB { get; set; }
        }

        private class TransformedTextData
        {
            public float[] FeaturesA { get; set; }
            public float[] FeaturesB { get; set; }
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
    }
}