using Microsoft.Extensions.Logging;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Image;
using TensorStack.Python;
using TensorStack.Python.Common;
using TensorStack.Python.Config;
using TensorStack.Python.Scheduler;

namespace DebugApp
{
    internal class Program
    {

        //static async Task Main(string[] args)
        //{
        //    ILoggerFactory loggerFactory = LoggerFactory.Create(builder =>
        //    {
        //        builder.AddConsole();
        //        builder.SetMinimumLevel(LogLevel.Trace);
        //    });
        //    ILogger logger = loggerFactory.CreateLogger<Program>();


        //    // Virtual Environment Config
        //    var environmentConfig = EnvironmentConfig.DefaultCUDA with { Environment = "default-test" };

        //    // Pipeline Config
        //    var pipelineConfig = new PipelineConfig
        //    {
        //        ModelPath = "Qwen/Qwen-Image-Edit",
        //        Pipeline = "QwenImagePipeline",
        //        ProcessType = ProcessType.ImageEdit,
        //        DataType = DataType.Bfloat16,
        //        MemoryMode = MemoryModeType.OffloadCPU
        //    };

        //    var pipelineClientConfig = new ClientConfig
        //    {
        //        IsDebugMode = true,
        //        Environment = environmentConfig,
        //        ServerPath = "D:\\Repositories\\TensorStack-Private\\Apps\\AmuseStudio.Server\\bin\\Debug\\net10.0-windows10.0.17763.0\\",
        //    };

        //    // Create Pipeline
        //    using (var pythonPipeline = new PipelineClient(pipelineClientConfig, PipelineProgress.ConsoleCallback))
        //    {
        //        // Download/Load Model
        //        await pythonPipeline.LoadAsync(pipelineConfig);

        //        // Generate Option
        //        var options = new PipelineOptions
        //        {
        //            Prompt = "yarn art style",
        //            Steps = 30,
        //            Width = 1024,
        //            Height = 1024,
        //            GuidanceScale = 4f,
        //            Scheduler = SchedulerType.FlowMatchEuler,
        //            InputImage = new ImageInput("C:\\Users\\Administrator\\Pictures\\image-1393932493.png")
        //        };

        //        // Generate
        //        var response = await pythonPipeline.RunAsync(options);

        //        // Save Image
        //        await response
        //            .AsImageTensor()
        //            .SaveAsync("Result.png");

        //        // Generate
        //        response = await pythonPipeline.RunAsync(options);

        //        // Save Image
        //        await response
        //            .AsImageTensor()
        //            .SaveAsync("Result2.png");

        //        await pythonPipeline.UnloadAsync();
        //    }

        //}

        static async Task Main(string[] args)
        {
            ILoggerFactory loggerFactory = LoggerFactory.Create(builder =>
            {
                builder.AddConsole();
                builder.SetMinimumLevel(LogLevel.Trace);
            });
            ILogger logger = loggerFactory.CreateLogger<Program>();


            // Environment Config
            var environmentConfig = EnvironmentConfig.DefaultCUDA with { Environment = "default-test" };
            var pythonManager = new PythonManager(environmentConfig);
            await pythonManager.CreateAsync(EnvironmentMode.Create, PipelineProgress.ConsoleCallback);

            // Pipeline Config
            var pipelineConfig = new PipelineConfig
            {
                ModelPath = "FLUX.2-klein-4B",
                Pipeline = "Flux2KleinPipeline",
                Template = "Flux2Klein4B",
                ProcessType = ProcessType.TextToImage,
                DataType = DataType.Bfloat16,
                MemoryMode = MemoryModeType.Device,
                QuantType = QuantizationType.Q8Bit
            };

            // Create Pipeline
            using (var pythonPipeline = new PythonPipeline(pipelineConfig, PipelineProgress.ConsoleCallback))
            {
                // Download/Load Model
                await pythonPipeline.LoadAsync();

                // Generate Option
                var options = new PipelineOptions
                {
                    Prompt = "Cat",
                    Seed = 123456,
                    Steps = 4,
                    Width = 1024,
                    Height = 1024,
                    GuidanceScale = 0f,
                    SchedulerOptions = new FlowMatchEulerOptions
                    {
                        Shift = 3f,
                        UseDynamicShifting = true
                    }
                };

                // Generate
                var response = await pythonPipeline.GenerateAsync(options);

                // Save Image
                await response
                    .First()
                    .AsImageTensor()
                    .SaveAsync("Result.png");
            }
        }

        //private static async Task DemoUpscaleImage(ExecutionProvider provider)
        //{

        //    // Pipeline Configuration
        //    var config = new UpscalerConfig
        //    {
        //        Channels = 3,
        //        ScaleFactor = 4,
        //        Normalization = Normalization.ZeroToOne,
        //        ExecutionProvider = Provider.GetProvider(),
        //        Path = "M:\\Amuse3\\_uploaded\\Upscale-amuse\\RealESR-General-4x\\model.onnx"
        //    };

        //    //// Create pipeline
        //    //using (var pipeline = UpscalePipeline.Create(config))
        //    //{
        //    //    // Upscale Options
        //    //    var options = new UpscaleImageOptions
        //    //    {
        //    //        Image = new ImageInput("Input.png")
        //    //    };

        //    //    // Run Pipeline
        //    //    var result = await pipeline.RunAsync(options);

        //    //    // Save Result
        //    //    await result.SaveAsync("Output.png");
        //    //}





        //    //// Create Pipeline
        //    //using (var pipeline = UpscalePipeline.Create(config))
        //    //{
        //    //    // Upscale Options
        //    //    var options = new UpscaleVideoOptions
        //    //    {
        //    //        Video = new VideoInput("Input.mp4"),
        //    //    };

        //    //    // Run Pipeline
        //    //    var outputTensor = await pipeline.RunAsync(options);

        //    //    // Save Output
        //    //    await outputTensor.SaveAync("Output.mp4");
        //    //}



        //    // Create Pipelines
        //    using (var upscalePipeline = UpscalePipeline.Create(config))
        //    using (var interpolationPipeline = InterpolationPipeline.Create(provider))
        //    {
        //        // Read Stream  [512 x 512 @ 8fps]
        //        var videoInput = new VideoInputStream("Input.mp4"); // 512 x 512 @ 8fps
        //        var videoStream = videoInput.GetAsync();

        //        // Upscale Stream
        //        videoStream = upscalePipeline.RunAsync(new UpscaleStreamOptions
        //        {
        //            Stream = videoStream
        //        });

        //        // Interpolate Stream
        //        videoStream = interpolationPipeline.RunAsync(new InterpolationStreamOptions
        //        {
        //            Multiplier = 3,
        //            Stream = videoStream,
        //            FrameRate = videoInput.FrameRate,
        //            FrameCount = videoInput.FrameCount
        //        });

        //        // Save Steam  [2048 x 2048 @ 24fps]
        //        await videoStream.SaveAync("Output.mp4");
        //    }
        //}

        //private static async Task DemoPhi3(ExecutionProvider provider)
        //{
        //    // Phi3
        //    System.Console.WriteLine($"\n\n----------------Phi3----------------");
        //    var phi3_Path = "M:\\BaseModels\\Phi-3-mini-4k-instruct-onnx\\cuda\\cuda-fp16";
        //    var phi3_pipeline = Phi3Pipeline.Create(provider, phi3_Path, PhiType.Mini, decoderModel: "phi3-mini-4k-instruct-cuda-fp16.onnx");
        //    var phi3_searchOptions = new GenerateOptions
        //    {
        //        Seed = 0,
        //        TopK = 50,
        //        Beams = 3,
        //        TopP = 0.9f,
        //        Temperature = 1f,
        //        LengthPenalty = -1f,
        //        DiversityLength = 20,
        //        NoRepeatNgramSize = 3,
        //        MinLength = 20,
        //        MaxLength = 4096,
        //        EarlyStopping = EarlyStopping.None,
        //        Prompt = "<|user|>What is an apple?<|end|><|assistant|>"
        //    };

        //    var onProgressUpdate = new Progress<GenerateProgress>((progress) =>
        //    {
        //        if (progress.IsReset)
        //            System.Console.WriteLine("\n\n-----------------------Beam Reset--------------------------");

        //        System.Console.Write(progress.Result);
        //    });

        //    var timestamp = Stopwatch.GetTimestamp();
        //    //foreach (var result in await phi3_pipeline.RunAsync(phi3_searchOptions, onProgressUpdate))
        //    //{
        //    //    System.Console.WriteLine("\n\n---------------------Result--------------------------");
        //    //    System.Console.WriteLine(result.Result);
        //    //    break;
        //    //    //System.Console.WriteLine("-----------------------------------------------------");
        //    //}


        //    var result = await phi3_pipeline.RunAsync(phi3_searchOptions, onProgressUpdate);
        //    System.Console.WriteLine("\n\n---------------------Result--------------------------"); ;
        //    System.Console.WriteLine(result.Result);
        //    System.Console.WriteLine();
        //    System.Console.WriteLine($"\nElapsed: {Stopwatch.GetElapsedTime(timestamp)}");
        //}

        //private static async Task DemoWhisper(ExecutionProvider provider)
        //{
        //    //// Whisper
        //    System.Console.WriteLine($"\n\n----------------Whisper----------------");
        //    var whisper_Path = "M:\\Models\\TensorStudio\\Models\\Audio\\Whisper-Base";
        //    var whisper_pipeline = WhisperPipeline.Create(provider, whisper_Path, WhisperType.Base);
        //    var whisper_searchOptions = new WhisperSearchOptions
        //    {
        //        Seed = -1,
        //        TopK = 50,
        //        Beams = 3,
        //        TopP = 0.9f,
        //        Temperature = 1f,
        //        LengthPenalty = 0f,
        //        DiversityLength = 448,
        //        NoRepeatNgramSize = 3,
        //        MinLength = 10,
        //        MaxLength = 448,
        //        EarlyStopping = EarlyStopping.None,

        //        Task = TextGeneration.Pipelines.Whisper.TaskType.Transcribe,
        //        Language = LanguageType.EN,
        //        AudioInput = await AudioInput.CreateAsync("kennedy.wav")
        //    };

        //    var onProgressUpdate = new Progress<GenerateProgress>((progress) =>
        //    {
        //        if (progress.IsReset)
        //            System.Console.WriteLine("\n\n-----------------------Beam Reset--------------------------");

        //        System.Console.Write(progress.Result);
        //    });

        //    var timestamp = Stopwatch.GetTimestamp();
        //    foreach (var result in await whisper_pipeline.RunAsync(whisper_searchOptions, onProgressUpdate))
        //    {
        //        System.Console.WriteLine();
        //        System.Console.WriteLine();
        //        System.Console.WriteLine(result.Result);
        //    }

        //    // var result = await whisper_pipeline.RunAsync(whisper_searchOptions);
        //    System.Console.WriteLine($"\nElapsed: {Stopwatch.GetElapsedTime(timestamp)}");
        //}

        //private static async Task DemoSummary(ExecutionProvider provider)
        //{
        //    // text_summarization
        //    System.Console.WriteLine($"------------Text Summary------------");
        //    var text_Path = "M:\\BaseModels\\text_summarization\\onnx";
        //    var text_pipeline = SummaryPipeline.Create(provider, text_Path);
        //    var text_searchOptions = new GenerateOptions
        //    {
        //        MaxLength = 512,
        //        Temperature = 0.9f,
        //        NoRepeatNgramSize = 4,
        //        Prompt = "summarize: The fine-tuned model presented here is an enhanced iteration of the DistilBERT-base-uncased model, meticulously trained on an updated dataset. Leveraging the underlying architecture of DistilBERT, a compact variant of BERT optimized for efficiency, this model is tailor-made for natural language processing tasks with a primary focus on question answering. Its training involved exposure to a diverse and contemporary dataset, ensuring its adaptability to a wide range of linguistic nuances and semantic intricacies. The fine-tuning process refines the model's understanding of context, allowing it to excel in tasks that require nuanced comprehension and contextual reasoning, making it a robust solution for question and answering applications in natural language processing."
        //    };

        //    var result = await text_pipeline.RunAsync(text_searchOptions);
        //    System.Console.WriteLine($"[{result.Beam}][{result.Score:F5}] - {result.Result}");
        //}

        //private static async Task DemoInterpolate(ExecutionProvider provider)
        //{
        //    // Create Pipeline
        //    using (var pipeline = InterpolationPipeline.Create(provider))
        //    {
        //        // Read Steam
        //        var inputStream = new VideoInputStream("Input.gif");

        //        // Interpolate stream
        //        var outputStream = pipeline.RunAsync(new InterpolationStreamOptions
        //        {
        //            Multiplier = 3,
        //            FrameRate = inputStream.FrameRate,
        //            FrameCount = inputStream.FrameCount,
        //            Stream = inputStream.GetAsync()
        //        });

        //        // Save Steam
        //        await outputStream.SaveAync("Output.mp4");
        //    }
        //}

        //private static async Task DemoPose(ExecutionProvider provider)
        //{
        //    // Create Pipeline
        //    var config = new ExtractorConfig
        //    {
        //        SampleSize = 640,
        //        ExecutionProvider = provider,
        //        Path = "M:\\rtmo_l_640x640_body7.onnx"
        //    };

        //    using (var pipeline = PosePipeline.Create(config))
        //    {
        //        var options = new PoseImageOptions
        //        {
        //            Detections = 2,
        //            BodyConfidence = 0,
        //            JointConfidence = 0.2f,
        //            Image = new ImageInput("image-866057774.png")
        //        };

        //        // Run Pipeline
        //        var result = await pipeline.RunAsync(options);

        //        // Save Result
        //        await result.SaveAsync("Output.png");

        //        //var videoInput = new VideoInputStream("5385814-sd_540_960_25fps.mp4"); // 512 x 512 @ 8fps
        //        //var videoStream = videoInput.GetAsync();

        //        ////Upscale Stream
        //        //videoStream = pipeline.RunAsync(new PoseStreamOptions
        //        //{
        //        //    BodyConfidence = 0.4f,
        //        //    JointConfidence = 0.3f,
        //        //    Stream = videoStream
        //        //});

        //        //await videoStream.SaveAync("Output.mp4");
        //    }
        //}

        //private static async Task DemoLlama(ExecutionProvider provider)
        //{
        //    // Phi3
        //    System.Console.WriteLine($"\n\n----------------Llama----------------");
        //    var path = "E:\\Qwen2.5-0.5B";
        //    var pipeline = LlamaPipeline.Create(provider, path);
        //    var searchOptions = new GenerateOptions
        //    {
        //        Seed = 0,
        //        TopK = 50,
        //        Beams = 3,
        //        TopP = 1f,
        //        Temperature = 1f,
        //        LengthPenalty = -1f,
        //        DiversityLength = 20,
        //        NoRepeatNgramSize = 3,
        //        MinLength = 20,
        //        MaxLength = 4096,
        //        EarlyStopping = EarlyStopping.None,
        //        Prompt = "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nWhat is large language model?<|im_end|>\n<|im_start|>assistant\n"
        //    };

        //    var onProgressUpdate = new Progress<GenerateProgress>((progress) =>
        //    {
        //        if (progress.IsReset)
        //            System.Console.WriteLine("\n\n-----------------------Beam Reset--------------------------");

        //        System.Console.Write(progress.Result);
        //    });

        //    var timestamp = Stopwatch.GetTimestamp();
        //    //foreach (var result in await phi3_pipeline.RunAsync(phi3_searchOptions, onProgressUpdate))
        //    //{
        //    //    System.Console.WriteLine("\n\n---------------------Result--------------------------");
        //    //    System.Console.WriteLine(result.Result);
        //    //    break;
        //    //    //System.Console.WriteLine("-----------------------------------------------------");
        //    //}



        //    var result = await pipeline.RunAsync(searchOptions, onProgressUpdate);
        //    System.Console.WriteLine("\n\n---------------------Result--------------------------"); ;
        //    System.Console.WriteLine(result.Result);
        //    System.Console.WriteLine();
        //    System.Console.WriteLine($"\nElapsed: {Stopwatch.GetElapsedTime(timestamp)}");
        //}




        //private static ExecutionProvider CreateProvider(int deviceId, GraphOptimizationLevel optimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL)
        //{
        //    OrtMemoryInfo memoryInfo = new OrtMemoryInfo(OrtMemoryInfo.allocatorCPU, OrtAllocatorType.DeviceAllocator, deviceId, OrtMemType.Default);
        //    return new ExecutionProvider("DMLExecutionProvider",  delegate
        //    {
        //        SessionOptions sessionOptions = new SessionOptions();
        //        sessionOptions.GraphOptimizationLevel = optimizationLevel;
        //        sessionOptions.AppendExecutionProvider_DML(deviceId);
        //        sessionOptions.AppendExecutionProvider_CPU();
        //        return sessionOptions;
        //    });
        //}

        //private static async Task TextToImage(NitroPipeline pipeline)
        //{
        //    System.Console.WriteLine("TextToImage...");
        //    var options = pipeline.DefaultOptions with
        //    {
        //        Seed = 624461087,
        //        GuidanceScale = 0,
        //        Prompt = "cute cat",
        //        //  NegativePrompt = "painting, drawing, sketches, monochrome, grayscale, illustration, anime, cartoon, graphic, text, crayon, graphite, abstract, easynegative, low quality, normal quality, worst quality, lowres, close up, cropped, out of frame, jpeg artifacts, duplicate, morbid, mutilated, mutated hands, poorly drawn hands, poorly drawn face, mutation, deformed, blurry, glitch, deformed, mutated, cross-eyed, ugly, dehydrated, bad anatomy, bad proportions, gross proportions, cloned face, disfigured, malformed limbs, missing arms, missing legs fused fingers, too many fingers,extra fingers, extra limbs,, extra arms, extra legs,disfigured,",
        //        IsLowMemoryEnabled = true,
        //        Scheduler = SchedulerType.FlowMatchEuler
        //    };

        //    var test = await pipeline.RunAsync(options);

        //    await test.SaveAsync("Test\\TextToImage.png");
        //}


        //private static async Task ImageToImage(StableDiffusionPipeline pipeline)
        //{
        //    System.Console.WriteLine("ImageToImage...");
        //    var inputImage = new ImageInput("M:\\Amuse3\\_uploaded\\StableDiffusion-amuse\\Sample.png", 512, 512);
        //    var options = pipeline.DefaultOptions with
        //    {
        //        Seed = 1234,
        //        Prompt = "party ballons",
        //        InputImage = inputImage,
        //        Strength = 0.7f
        //    };

        //    var test = await pipeline.RunAsync(options);

        //    await test.SaveAsync("M:\\Amuse3\\_uploaded\\StableDiffusion-amuse\\ImageToImage.png");
        //}



        //private static async Task ControlNet(StableDiffusionPipeline pipeline, ControlNetModel controlNet)
        //{
        //    System.Console.WriteLine("ControlNet...");
        //    var inputImage = new ImageInput("M:\\Amuse3\\_uploaded\\ControlNet-amuse\\StableDiffusion\\Depth\\Sample.png", 512, 512);
        //    var options = pipeline.DefaultOptions with
        //    {
        //        Seed = 1234,
        //        Prompt = "(cool:1.6) car",
        //        InputControlImage = inputImage,
        //        ControlNet = controlNet
        //    };

        //    var test = await pipeline.RunAsync(options);

        //    await test.SaveAsync("M:\\Amuse3\\_uploaded\\StableDiffusion-amuse\\ControlNet.png");
        //}


        //private static async Task ControlNetImage(StableDiffusionPipeline pipeline, ControlNetModel controlNet)
        //{
        //    System.Console.WriteLine("ControlNetImage...");
        //    var controlImage = new ImageInput("M:\\Amuse3\\_uploaded\\ControlNet-amuse\\StableDiffusion\\Depth\\Sample.png", 512, 512);
        //    var inputImage = new ImageInput("M:\\Amuse3\\_uploaded\\ControlNet-amuse\\StableDiffusion\\Depth\\Sample2.png", 512, 512);
        //    var options = pipeline.DefaultOptions with
        //    {
        //        Seed = 1234,
        //        Prompt = "(cool:1.6) car",
        //        InputImage = inputImage,
        //        InputControlImage = controlImage,
        //        ControlNet = controlNet,
        //        Strength = 0.3f
        //    };

        //    var test = await pipeline.RunAsync(options);

        //    await test.SaveAsync("M:\\Amuse3\\_uploaded\\StableDiffusion-amuse\\ControlNetImage.png");
        //}

        //[nuget: TensorStack.Video.Windows]
        //[nuget: TensorStack.Providers.DML]

        //async Task QuickStartAsync()
        //{
        //    var provider = Provider.GetProvider();

        //    // Create Pipeline
        //    using (var pipeline = InterpolationPipeline.Create(provider))
        //    {
        //        // Read Steam
        //        var inputStream = new VideoInputStream("Input.mp4");

        //        // Interpolate stream
        //        var outputStream = pipeline.RunAsync(new InterpolationStreamOptions
        //        {
        //            Multiplier = 3,
        //            Stream = inputStream
        //        });

        //        // Save Steam
        //        await outputStream.SaveAync("Output.mp4");
        //    }
        //}
    }

    public static class Ext
    {

    }
}
