using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using Microsoft.ML;
using Microsoft.ML.Transforms.Text;
using NPOI.SS.UserModel;
using NPOI.XSSF.UserModel;

namespace CosineCalculationProjectOptimizations
{
    public static class ApplyWordEmbedding
    {
        public static void Main(string[] args)
        {
            var stopwatchWholeProcess = new Stopwatch();
            stopwatchWholeProcess.Start();
            Example();
            stopwatchWholeProcess.Stop();
            Console.WriteLine($"Elapsed time for whole process: {stopwatchWholeProcess.ElapsedMilliseconds} ms.");
        }

        public static void Example()
        {
            var mlContext = new MLContext();

            var filePath = "C:\\Users\\Igor\\Desktop\\models\\наборы данных\\En_Es_Deepl_Kefir_Grim_Soul\\Kefir — Grim Soul_translation_memory_en_and_es_translation_and_es_deepl_only_text.xlsx";
            var textDataItems = LoadDataFromExcel(filePath);

            var emptyTextDataSamples = new TextDataItem[textDataItems.Length];
            var emptyTextDataView = mlContext.Data.LoadFromEnumerable(emptyTextDataSamples);

            var textPipeline = mlContext.Transforms.Text.NormalizeText("Text")
                .Append(mlContext.Transforms.Text.TokenizeIntoWords("Tokens", "Text"))
                .Append(mlContext.Transforms.Text.ApplyWordEmbedding("Features", "Tokens", WordEmbeddingEstimator.PretrainedModelKind.GloVeTwitter25D));

            var textTransformer = textPipeline.Fit(emptyTextDataView);

            var textPredictionEngine = mlContext.Model.CreatePredictionEngine<TextDataItem, TransformedTextData>(textTransformer);

            var stopwatchCalculationVectorization = new Stopwatch();
            stopwatchCalculationVectorization.Start();
            for (int i = 0; i < textDataItems.Length; i++)
            {
                var prediction = textPredictionEngine.Predict(textDataItems[i]);
                if (prediction != null && prediction.Features != null)
                {
                    textDataItems[i].Features = prediction.Features;
                }
            }
            stopwatchCalculationVectorization.Stop();

            var stopwatchCalculationCosineSimilarityWithStableCount = new Stopwatch();
            stopwatchCalculationCosineSimilarityWithStableCount.Start();

            int itemCount = textDataItems.Length;
            Parallel.For(0, itemCount, i =>
            {
                for (int j = i + 1; j < itemCount; j++)
                {
                    var cosineSimilarity = CalculateCosineSimilarity(textDataItems[i].Features, textDataItems[j].Features);
                }
            });
            stopwatchCalculationCosineSimilarityWithStableCount.Stop();

            Console.WriteLine($"Elapsed time for calculation vectorization for {textDataItems.Length} strings: {stopwatchCalculationVectorization.ElapsedMilliseconds} ms.");
            Console.WriteLine($"Elapsed time for calculation cosine similarity with const itemCount for {textDataItems.Length} strings: {stopwatchCalculationCosineSimilarityWithStableCount.ElapsedMilliseconds} ms.");
            Console.WriteLine($"TextDataItem array length is {textDataItems.Length}");
        }

        private static TextDataItem[] LoadDataFromExcel(string filePath)
        {
            var textDataItems = new List<TextDataItem>();

            using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read))
            {
                var workbook = new XSSFWorkbook(fileStream);
                var sheet = workbook.GetSheetAt(0);

                for (int i = 1; i <= sheet.LastRowNum; i++)
                {
                    var row = sheet.GetRow(i);
                    if (row == null)
                        continue;

                    var textA = row.GetCell(0)?.StringCellValue;

                    for (int j = 1; j < row.LastCellNum; j++)
                    {
                        var textDataItem = new TextDataItem { Text = textA, RowNumber = i };
                        textDataItems.Add(textDataItem);
                    }
                }
            }

            return textDataItems.ToArray();
        }

        private class TextDataItem
        {
            public string Text { get; set; }
            public int RowNumber { get; set; }
            public float[] Features { get; set; }
        }

        private class TransformedTextData
        {
            public float[] Features { get; set; }
        }

        private static float CalculateCosineSimilarity(float[] vector1, float[] vector2)
        {
            if (vector1 != null && vector2 != null && vector1.Length == vector2.Length)
            {
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

            return 0.0f;
        }
    }
}