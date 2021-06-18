using System;
using System.IO;
using Microsoft.ML;
using SentimentAnalysis.Models;

namespace SentimentAnalysis
{
    internal static class Program
    {
        private static readonly string DataPath = Path.Combine(Directory.GetCurrentDirectory(), "Data", "wikiDetoxAnnotated40kRows.tsv");
        
        private static void Main(string[] args)
        {
            var mlContext = new MLContext();
            
            //Загружаем все данные из TSV файла.
            var dataView = mlContext.Data.LoadFromTextFile<SentimentIssue>(DataPath, hasHeader: true);

            //Разделаем данные: 80% - для обучения и 20% для теста. 
            var trainTestSplit = mlContext.Data.TrainTestSplit(dataView, 0.2);
            //Данные для обучения.
            var trainingData = trainTestSplit.TrainSet;
            //Данные для теста.
            var testData = trainTestSplit.TestSet;

            //Определяем процесс обучения модели.
            //Features - данные, определяющие Label, факторы, влияющие на результат.
            //Label - результат, определяемый Features.
            var dataProcessPipeline = mlContext.Transforms.Text
                .FeaturizeText("Features", nameof(SentimentIssue.Text));

            var trainer = mlContext.BinaryClassification.Trainers.SdcaLogisticRegression();
            var trainingPipeline = dataProcessPipeline.Append(trainer);

            Console.WriteLine("Обучение модели...");
            //Обучение модели.
            var trainedModel = trainingPipeline.Fit(trainingData);
            Console.WriteLine("Обучение завершено.");
            
            var predictions = trainedModel.Transform(testData);
            //Проходимся по данным для теста.
            var metrics = mlContext.BinaryClassification.Evaluate(predictions);
            //Процент точности модели.
            var accuracyPercent = metrics.Accuracy * 100D;
            Console.WriteLine($"Accuracy: {accuracyPercent:N2}%");

            //API для предугадывания результата.
            var predicationEngine =
                mlContext.Model.CreatePredictionEngine<SentimentIssue, SentimentPrediction>(trainedModel);

            //0	555501737.0	"  Jack, is it a misrepresentation for you to write more than 500,000 nucleotide base pairs, or should you write 582 base pair.   Lets start watching ""RNA world"" page too.   "	2013	True	user	blocked	train

            void PrintPredication(string commentText) => Program.PrintPredication(predicationEngine, commentText);
            
            PrintPredication("Hello, my name is John");
            PrintPredication("F*ck you, Johny!");
            PrintPredication("Free access for article citation/reference purposes?");
            PrintPredication("Just$@#@#@% looked and timko info as been removed bu YOU fuck off");
            PrintPredication("\"Jonathan is Offline\" - Jonathans Talk Page  !!@$@$ f");
            PrintPredication("Are you a female????? Get out of here");
        }

        private static void PrintPredication(PredictionEngine<SentimentIssue, SentimentPrediction> predictionEngine, string commentText)
        {
            //Получаем предполагаемый моделью результат.
            var resultPredication = predictionEngine.Predict(new SentimentIssue
            {
                Text = commentText
            });
            
            Console.WriteLine($"{commentText}\n-> {(resultPredication.Prediction ? "Негативный" : "Положительный")}\n");
        }
    }
}