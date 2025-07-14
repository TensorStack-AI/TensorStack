// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// TransformerModel: Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
    /// </summary>
    public abstract class TransformerModel : IDisposable
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerModel(TransformerModelConfig configuration)
        {
            Configuration = configuration;
            Transformer = new ModelSession<TransformerModelConfig>(configuration);
            if (!string.IsNullOrEmpty(configuration.ControlNetPath))
                TransformerControlNet = new ModelSession<TransformerModelConfig>(configuration with { Path = configuration.ControlNetPath });
        }

        /// <summary>
        /// Gets the configuration.
        /// </summary>
        protected TransformerModelConfig Configuration { get; }

        /// <summary>
        /// Gets the Transformer.
        /// </summary>
        protected ModelSession<TransformerModelConfig> Transformer { get; }

        /// <summary>
        /// Gets the Transformer ControlNet.
        /// </summary>
        protected ModelSession<TransformerModelConfig> TransformerControlNet { get; }

        /// <summary>
        /// Gets the type of the model.
        /// </summary>
        public ModelType ModelType => Configuration.ModelType;

        /// <summary>
        /// Gets the in channels.
        /// </summary>
        public int InChannels => Configuration.InChannels;

        /// <summary>
        /// Gets the out channels.
        /// </summary>
        public int OutChannels => Configuration.OutChannels;

        /// <summary>
        /// Gets the joint attention.
        /// </summary>
        public int JointAttention => Configuration.JointAttention;

        /// <summary>
        /// Gets the pooled projection.
        /// </summary>
        public int PooledProjection => Configuration.PooledProjection;

        /// <summary>
        /// Gets the caption projection.
        /// </summary>
        public int CaptionProjection => Configuration.CaptionProjection;

        /// <summary>
        /// Gets a value indicating whether this instance has TransformerControlNet.
        /// </summary>
        public bool HasControlNet => TransformerControlNet != null;


        /// <summary>
        /// Determines whether Transformer is loaded.
        /// </summary>
        public bool IsLoaded()
        {
            return Transformer.IsLoaded();
        }


        /// <summary>
        /// Determines whether TransformerControlNet is loaded.
        /// </summary>
        public bool IsControlNetLoaded()
        {
            return TransformerControlNet.IsLoaded();
        }


        /// <summary>
        /// Load Transformer model
        /// </summary>
        /// <param name="onnxOptimizations">The onnx optimizations.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ModelMetadata> LoadAsync(ModelOptimization optimizations = null, CancellationToken cancellationToken = default)
        {
            await UnloadControlNetAsync();
            return await Transformer.LoadAsync(optimizations, cancellationToken);
        }


        /// <summary>
        /// Load TransformerControlNet model
        /// </summary>
        /// <param name="onnxOptimizations">The onnx optimizations.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ModelMetadata> LoadControlNetAsync(ModelOptimization optimizations = null, CancellationToken cancellationToken = default)
        {
            await UnloadAsync();
            return await TransformerControlNet.LoadAsync(optimizations, cancellationToken);
        }


        /// <summary>
        /// Unload Transformer model
        /// </summary>
        public async Task UnloadAsync()
        {
            if (Transformer.IsLoaded())
                await Transformer.UnloadAsync();
        }


        /// <summary>
        /// Unload TransformerControlNet model
        /// </summary>
        public async Task UnloadControlNetAsync()
        {
            if (TransformerControlNet.IsLoaded())
                await TransformerControlNet.UnloadAsync();
        }


        /// <summary>
        /// Determines whether Transformer optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        public bool HasOptimizationsChanged(ModelOptimization optimizations)
        {
            return Transformer.HasOptimizationsChanged(optimizations);
        }


        /// <summary>
        /// Determines whether TransformerControlNet optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        public bool HasControlNetOptimizationsChanged(ModelOptimization optimizations)
        {
            return TransformerControlNet.HasOptimizationsChanged(optimizations);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Transformer?.Dispose();
            TransformerControlNet?.Dispose();
            GC.SuppressFinalize(this);
        }

    }
}
