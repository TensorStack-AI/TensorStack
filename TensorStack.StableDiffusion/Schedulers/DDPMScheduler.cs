// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Schedulers
{
    public class DDPMScheduler : SchedulerBase
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="DDPMScheduler"/> class.
        /// </summary>
        /// <param name="options">The scheduler options.</param>
        public DDPMScheduler(ISchedulerOptions options) : base(options) { }


        /// <summary>
        /// Sets the timesteps.
        /// </summary>
        /// <returns></returns>
        protected override int[] SetTimesteps()
        {
            var timesteps = CreateTimestepSpacing();
            return timesteps
                 .Select(x => (int)Math.Round(x))
                 .OrderByDescending(x => x)
                 .ToArray();
        }


        /// <summary>
        /// Scales the input.
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        public override Tensor<float> ScaleInput(int timestep, Tensor<float> sample)
        {
            return sample;
        }


        /// <summary>
        /// Computes the next prediction steps
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="sample">The sample.</param>
        /// <param name="previousSample">The previous sample.</param>
        /// <returns>SchedulerResult.</returns>
        public override SchedulerResult Step(int timestep, Tensor<float> sample, Tensor<float> previousSample)
        {
            CurrentStep++;
            int currentTimestep = timestep;
            int currentTimestepIndex = Timesteps.IndexOf(currentTimestep);
            int previousTimestepIndex = currentTimestepIndex + 1;
            int previousTimestep = Timesteps.ElementAtOrDefault(previousTimestepIndex);

            //# 1. compute alphas, betas
            float alphaProdT = AlphasCumProd[currentTimestep];
            float alphaProdTPrev = previousTimestep >= 0 ? AlphasCumProd[previousTimestep] : 1f;
            float betaProdT = 1f - alphaProdT;
            float betaProdTPrev = 1f - alphaProdTPrev;
            float currentAlphaT = alphaProdT / alphaProdTPrev;
            float currentBetaT = 1f - currentAlphaT;
            float predictedVariance = 0;

            // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L390
            //if (modelOutput.Dimensions[1] == sample.Dimensions[1] * 2 && VarianceTypeIsLearned())
            //{
            //    DenseTensor<float>[] splitModelOutput = modelOutput.Split(modelOutput.Dimensions[1] / 2, 1);
            //    TensorHelper.SplitTensor(modelOutput, )
            //    modelOutput = splitModelOutput[0];
            //    predictedVariance = splitModelOutput[1];
            //}


            //# 2. compute predicted original sample from predicted noise also called
            //# "predicted x_0" of formula (15) from https://arxiv.org/pdf/2006.11239.pdf
            var predOriginalSample = GetPredictedSample(sample, previousSample, alphaProdT, betaProdT);

            //# 3. Clip or threshold "predicted x_0"
            if (Options.Thresholding)
            {
                // TODO: https://github.com/huggingface/diffusers/blob/main/src/diffusers/schedulers/scheduling_ddpm.py#L322
                // predOriginalSample = ThresholdSample(predOriginalSample);
            }
            else if (Options.ClipSample)
            {
                predOriginalSample = predOriginalSample.ClipTo(-Options.ClipSampleRange, Options.ClipSampleRange);
            }

            //# 4. Compute coefficients for pred_original_sample x_0 and current sample x_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            float predOriginalSampleCoeff = (float)Math.Sqrt(alphaProdTPrev) * currentBetaT / betaProdT;
            float currentSampleCoeff = (float)Math.Sqrt(currentAlphaT) * betaProdTPrev / betaProdT;


            //# 5. Compute predicted previous sample µ_t
            //# See formula (7) from https://arxiv.org/pdf/2006.11239.pdf
            //pred_prev_sample = pred_original_sample_coeff * pred_original_sample + current_sample_coeff * sample
            var predPrevSample = previousSample
                .MultiplyTo(currentSampleCoeff)
                .AddTo(predOriginalSample.MultiplyTo(predOriginalSampleCoeff));


            //# 6. Add noise
            if (currentTimestep > 0)
            {
                Tensor<float> variance;
                var varianceNoise = CreateRandomSample(sample.Dimensions);
                if (Options.VarianceType == VarianceType.FixedSmallLog)
                {
                    var v = GetVariance(currentTimestep, predictedVariance);
                    variance = varianceNoise.MultiplyTo(v);
                }
                else if (Options.VarianceType == VarianceType.LearnedRange)
                {
                    var v = (float)Math.Exp(0.5 * GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultiplyTo(v);
                }
                else
                {
                    var v = (float)Math.Sqrt(GetVariance(currentTimestep, predictedVariance));
                    variance = varianceNoise.MultiplyTo(v);
                }
                predPrevSample = predPrevSample.AddTo(variance);
            }

            return new SchedulerResult(predPrevSample);
        }


        /// <summary>
        /// Adds noise to the sample.
        /// </summary>
        /// <param name="timesteps">The timesteps.</param>
        /// <param name="sample">The original sample.</param>
        /// <param name="noise">The noise.</param>
        public override Tensor<float> ScaleNoise(int timestep, Tensor<float> sample, Tensor<float> noise)
        {
            float alphaProd = AlphasCumProd[timestep];
            float sqrtAlpha = MathF.Sqrt(alphaProd);
            float sqrtOneMinusAlpha = MathF.Sqrt(1.0f - alphaProd);
            return noise
                .MultiplyTo(sqrtOneMinusAlpha)
                .AddTo(sample.MultiplyTo(sqrtAlpha));
        }


        /// <summary>
        /// Gets the predicted sample.
        /// </summary>
        /// <param name="sample">The model output.</param>
        /// <param name="previousSample">The sample.</param>
        /// <param name="alphaProdT">The alpha product t.</param>
        /// <param name="betaProdT">The beta product t.</param>
        private Tensor<float> GetPredictedSample(Tensor<float> sample, Tensor<float> previousSample, float alphaProdT, float betaProdT)
        {
            Tensor<float> predictedSample = null;
            if (Options.PredictionType == PredictionType.Epsilon)
            {
                //pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
                var sampleBeta = previousSample.SubtractTo(sample.MultiplyTo((float)Math.Sqrt(betaProdT)));
                predictedSample = sampleBeta.DivideTo((float)Math.Sqrt(alphaProdT));
            }
            else if (Options.PredictionType == PredictionType.Sample)
            {
                predictedSample = sample;
            }
            else if (Options.PredictionType == PredictionType.VariablePrediction)
            {
                // pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
                var alphaSqrt = (float)Math.Sqrt(alphaProdT);
                var betaSqrt = (float)Math.Sqrt(betaProdT);
                predictedSample = previousSample
                    .MultiplyTo(alphaSqrt)
                    .SubtractTo(sample.MultiplyTo(betaSqrt));
            }
            return predictedSample;
        }


        /// <summary>
        /// Gets the variance.
        /// </summary>
        /// <param name="timestep">The t.</param>
        /// <param name="predictedVariance">The predicted variance.</param>
        /// <returns></returns>
        private float GetVariance(int timestep, float predictedVariance = 0f)
        {
            int prevTimestep = GetPreviousTimestep(timestep);
            float alphaProdT = AlphasCumProd[timestep];
            float alphaProdTPrev = prevTimestep >= 0 ? AlphasCumProd[prevTimestep] : 1.0f;
            float currentBetaT = 1 - alphaProdT / alphaProdTPrev;

            // For t > 0, compute predicted variance βt
            float variance = (1 - alphaProdTPrev) / (1 - alphaProdT) * currentBetaT;

            // Clamp variance to ensure it's not 0
            variance = Math.Max(variance, 1e-20f);


            if (Options.VarianceType == VarianceType.FixedSmallLog)
            {
                variance = (float)Math.Exp(0.5 * Math.Log(variance));
            }
            else if (Options.VarianceType == VarianceType.FixedLarge)
            {
                variance = currentBetaT;
            }
            else if (Options.VarianceType == VarianceType.FixedLargeLog)
            {
                variance = (float)Math.Log(currentBetaT);
            }
            else if (Options.VarianceType == VarianceType.Learned)
            {
                return predictedVariance;
            }
            else if (Options.VarianceType == VarianceType.LearnedRange)
            {
                float minLog = (float)Math.Log(variance);
                float maxLog = (float)Math.Log(currentBetaT);
                float frac = (predictedVariance + 1) / 2;
                variance = frac * maxLog + (1 - frac) * minLog;
            }
            return variance;
        }

    }
}
