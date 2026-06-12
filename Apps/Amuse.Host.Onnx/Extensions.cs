using Amuse.Common;
using System;
using System.IO;
using System.Linq;
using TensorStack.Common;
using TensorStack.StableDiffusion.Common;

namespace Amuse.Host.Onnx
{
    public static class Extensions
    {
        public static GenerateOptions ToOnnxOptions(this PipelineRunOptions options, PipelineLoadOptions loadOptions, ExecutionProvider executionProvider)
        {
            var generateOptions = new GenerateOptions
            {
                Width = options.Width,
                Height = options.Height,
                Prompt = options.Prompt,
                NegativePrompt = options.NegativePrompt,
                GuidanceScale = options.GuidanceScale,
                GuidanceScale2 = options.GuidanceScale2,
                ControlNetStrength = options.ControlNetScale,
                Seed = options.Seed,
                Steps = options.Steps,
                Steps2 = options.Steps2,
                Strength = options.Strength,
                InputImage = options.InputImages.FirstOrDefault(),
                InputControlImage = options.InputControlImages.FirstOrDefault(),
            };

            SetControlNet(executionProvider, loadOptions, generateOptions);
            SetMemoryOptions(loadOptions, generateOptions);
            SetSchedulerOptions(generateOptions, options.SchedulerOptions);
            return generateOptions;
        }


        public static TensorStack.TextGeneration.Pipelines.Whisper.LanguageType GetLanguageType(this PipelineRunOptions options)
        {
            if (Enum.TryParse<TensorStack.TextGeneration.Pipelines.Whisper.LanguageType>(options.Language.GetShortName(), true, out var languageType))
                return languageType;

            return TensorStack.TextGeneration.Pipelines.Whisper.LanguageType.EN;
        }


        private static void SetMemoryOptions(PipelineLoadOptions options, GenerateOptions generateOptions)
        {
            var memoryMode = options.MemoryMode;
            generateOptions.IsPipelineCacheEnabled = true;
            if (memoryMode == MemoryModeType.Device)
            {
                generateOptions.IsLowMemoryEnabled = false;
                generateOptions.IsLowMemoryComputeEnabled = false;
                generateOptions.IsLowMemoryDecoderEnabled = false;
                generateOptions.IsLowMemoryEncoderEnabled = false;
                generateOptions.IsLowMemoryTextEncoderEnabled = false;
            }
            else
            {
                generateOptions.IsLowMemoryEnabled = true;
                generateOptions.IsLowMemoryComputeEnabled = true;
                generateOptions.IsLowMemoryDecoderEnabled = true;
                generateOptions.IsLowMemoryEncoderEnabled = true;
                generateOptions.IsLowMemoryTextEncoderEnabled = true;
            }
        }


        private static void SetSchedulerOptions(GenerateOptions generateOptions, SchedulerOptions schedulerOptions)
        {
            generateOptions.AestheticNegativeScore = schedulerOptions.Shift;
            generateOptions.AestheticScore = schedulerOptions.Shift;
            generateOptions.BetaEnd = schedulerOptions.BetaEnd;
            generateOptions.BetaStart = schedulerOptions.BetaStart;
            generateOptions.SampleMaxValue = schedulerOptions.SampleMaxValue;
            generateOptions.Shift = schedulerOptions.Shift;
            generateOptions.StepsOffset = schedulerOptions.StepsOffset;
            generateOptions.Thresholding = schedulerOptions.Thresholding;
            generateOptions.TrainSteps = schedulerOptions.NumTrainTimesteps;
            generateOptions.UseKarrasSigmas = schedulerOptions.UseKarrasSigmas;
            // AlphaTransformType = ,
            // Timesteps = ,
            // TrainedBetas = ,
            // ClipSkip = ,
            // MaximumBeta = ,
            // AestheticNegativeScore = ,
            // AestheticScore = ,

            generateOptions.BetaSchedule = schedulerOptions.BetaSchedule switch
            {
                Amuse.Common.BetaScheduleType.Sigmoid => TensorStack.StableDiffusion.Enums.BetaScheduleType.Sigmoid,
                Amuse.Common.BetaScheduleType.SquaredCosine => TensorStack.StableDiffusion.Enums.BetaScheduleType.SquaredCosCapV2,
                Amuse.Common.BetaScheduleType.Linear => TensorStack.StableDiffusion.Enums.BetaScheduleType.Linear,
                Amuse.Common.BetaScheduleType.ScaledLinear => TensorStack.StableDiffusion.Enums.BetaScheduleType.ScaledLinear,
                _ => TensorStack.StableDiffusion.Enums.BetaScheduleType.ScaledLinear
            };

            generateOptions.PredictionType = schedulerOptions.PredictionType switch
            {
                Amuse.Common.PredictionType.Epsilon => TensorStack.StableDiffusion.Enums.PredictionType.Epsilon,
                Amuse.Common.PredictionType.Variable => TensorStack.StableDiffusion.Enums.PredictionType.VariablePrediction,
                Amuse.Common.PredictionType.Sample => TensorStack.StableDiffusion.Enums.PredictionType.Sample,
                _ => TensorStack.StableDiffusion.Enums.PredictionType.Epsilon
            };

            generateOptions.TimestepSpacing = schedulerOptions.TimestepSpacing switch
            {
                Amuse.Common.TimestepSpacingType.Linspace => TensorStack.StableDiffusion.Enums.TimestepSpacingType.Linspace,
                Amuse.Common.TimestepSpacingType.Trailing => TensorStack.StableDiffusion.Enums.TimestepSpacingType.Trailing,
                Amuse.Common.TimestepSpacingType.Leading => TensorStack.StableDiffusion.Enums.TimestepSpacingType.Leading,
                _ => TensorStack.StableDiffusion.Enums.TimestepSpacingType.Linspace
            };

            generateOptions.VarianceType = schedulerOptions.VarianceType switch
            {
                Amuse.Common.VarianceType.FixedSmallLog => TensorStack.StableDiffusion.Enums.VarianceType.FixedSmallLog,
                Amuse.Common.VarianceType.FixedSmall => TensorStack.StableDiffusion.Enums.VarianceType.FixedSmall,
                Amuse.Common.VarianceType.FixedLargeLog => TensorStack.StableDiffusion.Enums.VarianceType.FixedLargeLog,
                Amuse.Common.VarianceType.FixedLarge => TensorStack.StableDiffusion.Enums.VarianceType.FixedLarge,
                Amuse.Common.VarianceType.LearnedRange => TensorStack.StableDiffusion.Enums.VarianceType.LearnedRange,
                Amuse.Common.VarianceType.Learned => TensorStack.StableDiffusion.Enums.VarianceType.Learned,
                _ => TensorStack.StableDiffusion.Enums.VarianceType.FixedSmall
            };

            generateOptions.Scheduler = schedulerOptions.Scheduler switch
            {
                Amuse.Common.SchedulerType.DDIM => TensorStack.StableDiffusion.Enums.SchedulerType.DDIM,
                Amuse.Common.SchedulerType.DDPM => TensorStack.StableDiffusion.Enums.SchedulerType.DDPM,
                Amuse.Common.SchedulerType.DDPMWuerstchen => TensorStack.StableDiffusion.Enums.SchedulerType.DDPMWuerstchen,
                Amuse.Common.SchedulerType.Euler => TensorStack.StableDiffusion.Enums.SchedulerType.Euler,
                Amuse.Common.SchedulerType.EulerAncestral => TensorStack.StableDiffusion.Enums.SchedulerType.EulerAncestral,
                Amuse.Common.SchedulerType.FlowMatchEuler => TensorStack.StableDiffusion.Enums.SchedulerType.FlowMatchEulerDiscrete,
                Amuse.Common.SchedulerType.FlowMatchHeun => TensorStack.StableDiffusion.Enums.SchedulerType.FlowMatchEulerDynamic,
                Amuse.Common.SchedulerType.KDPM2 => TensorStack.StableDiffusion.Enums.SchedulerType.KDPM2,
                Amuse.Common.SchedulerType.KDPM2Ancestral => TensorStack.StableDiffusion.Enums.SchedulerType.KDPM2Ancestral,
                Amuse.Common.SchedulerType.LCM => TensorStack.StableDiffusion.Enums.SchedulerType.LCM,
                Amuse.Common.SchedulerType.LMS => TensorStack.StableDiffusion.Enums.SchedulerType.LMS,
                _ => TensorStack.StableDiffusion.Enums.SchedulerType.EulerAncestral
            };
        }


        private static void SetControlNet(ExecutionProvider executionProvider, PipelineLoadOptions options, GenerateOptions generateOptions)
        {
            if (options.ControlNet != null)
            {
                var modelPath = options.ControlNet.Path.EndsWith(".onnx") ? options.ControlNet.Path : Path.Combine(options.ControlNet.Path, "model.onnx");
                generateOptions.ControlNet = TensorStack.StableDiffusion.Models.ControlNetModel.FromFile(modelPath, executionProvider, options.ControlNet.Invert, options.ControlNet.LayerCount, options.ControlNet.DisableProjections);
            }
        }

    }
}
