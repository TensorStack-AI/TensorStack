// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Schedulers;

namespace TensorStack.StableDiffusion.Pipelines
{
    public abstract class PipelineBase : IDisposable
    {
        private GenerateOptions _defaultOptions;
        private IReadOnlyList<SchedulerType> _schedulers;

        /// <summary>
        /// Initializes a new instance of the <see cref="PipelineBase"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public PipelineBase(ILogger logger)
        {
            Logger = logger;
        }

        /// <summary>
        /// Gets the logger.
        /// </summary>
        protected ILogger Logger { get; }

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public abstract string Name { get; init; }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public abstract PipelineType PipelineType { get; }

        /// <summary>
        /// Gets the default scheduler options.
        /// </summary>
        public GenerateOptions DefaultOptions => _defaultOptions;

        /// <summary>
        /// Gets the pipelines supported schedulers.
        /// </summary>
        public IReadOnlyList<SchedulerType> Schedulers => _schedulers;

        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected abstract GenerateOptions ConfigureDefaultOptions();

        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected abstract IReadOnlyList<SchedulerType> ConfigureSchedulers();

        /// <summary>
        /// Checks the state of the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        protected abstract Task CheckPipelineState(IPipelineOptions options);

        /// <summary>
        /// Initializes this instance.
        /// </summary>
        protected void Initialize()
        {
            _schedulers = ConfigureSchedulers();
            _defaultOptions = ConfigureDefaultOptions();
        }


        /// <summary>
        /// Validates the options.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual void ValidateOptions(GenerateOptions options)
        {
            if (options.Seed <= 0)
                options.Seed = Random.Shared.Next();

            if (!Schedulers.Contains(options.Scheduler))
                throw new ArgumentException($"{options.Scheduler} not supported");

            if (options.HasControlNet && !options.HasInputControlImage)
                throw new ArgumentException("InputControlImage was not supplied");
        }


        /// <summary>
        /// Creates the scheduler.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual IScheduler CreateScheduler(GenerateOptions options)
        {
            IScheduler scheduler = options.Scheduler switch
            {
                SchedulerType.LMS => new LMSScheduler(options),
                SchedulerType.Euler => new EulerScheduler(options),
                SchedulerType.EulerAncestral => new EulerAncestralScheduler(options),
                SchedulerType.DDPM => new DDPMScheduler(options),
                SchedulerType.DDIM => new DDIMScheduler(options),
                SchedulerType.KDPM2 => new KDPM2Scheduler(options),
                SchedulerType.KDPM2Ancestral => new KDPM2AncestralScheduler(options),
                SchedulerType.DDPMWuerstchen => new DDPMWuerstchenScheduler(options),
                SchedulerType.LCM => new LCMScheduler(options),
                SchedulerType.FlowMatchEulerDiscrete => new FlowMatchEulerDiscreteScheduler(options),
                SchedulerType.FlowMatchEulerDynamic => new FlowMatchEulerDynamicScheduler(options),
                _ => default
            };
            scheduler.Initialize(options.Strength);
            return scheduler;
        }


        /// <summary>
        /// Applies the classifier-free guidance.
        /// </summary>
        /// <param name="prediction">The prediction.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        protected Tensor<float> ApplyGuidance(Tensor<float> prediction, float guidanceScale)
        {
            var length = (int)prediction.Length / 2;
            var conditional = prediction.Memory[length..];
            var unconditional = prediction.Memory[..length];
            unconditional.Lerp(conditional, guidanceScale);
            return new Tensor<float>(unconditional, [1, .. prediction.Dimensions.Slice(1)]);
        }


        /// <summary>
        /// Applies the classifier-free guidance.
        /// </summary>
        /// <param name="conditional">The conditional prediction.</param>
        /// <param name="unconditional">The unconditional prediction.</param>
        /// <param name="guidanceScale">The guidance scale.</param>
        protected Tensor<float> ApplyGuidance(Tensor<float> conditional, Tensor<float> unconditional, float guidanceScale)
        {
            unconditional.Memory.Lerp(conditional.Memory, guidanceScale);
            return unconditional;
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(disposing: true);
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        protected abstract void Dispose(bool disposing);
    }
}
