// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;

namespace TensorStack.Common
{
    /// <summary>
    /// ModelSession class to manage lifetime of an InferenceSession ans its SessionOptions.
    /// </summary>
    /// <typeparam name="T">ModelConfig implementation</typeparam>
    /// <seealso cref="IDisposable" />
    public class ModelSession<T> : IDisposable where T : ModelConfig
    {
        private readonly T _configuration;
        private ModelMetadata _metadata;
        private InferenceSession _session;
        private SessionOptions _options;
        private OrtAllocator _allocator;
        private ModelOptimization _optimizations;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public ModelSession(T configuration)
        {
            ArgumentNullException.ThrowIfNull(configuration.ExecutionProvider);
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException("Onnx model file not found, Path: {Path}", configuration.Path);

            _configuration = configuration;
        }


        /// <summary>
        /// Gets the metadata.
        /// </summary>
        public ModelMetadata Metadata => _metadata;

        /// <summary>
        /// Gets the SessionOptions.
        /// </summary>
        public SessionOptions Options => _options;

        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        public T Configuration => _configuration;

        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        public InferenceSession Session => _session;

        /// <summary>
        /// Gets the allocator.
        /// </summary>
        public OrtAllocator Allocator => _allocator;


        /// <summary>
        /// Loads the model session.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>ModelMetadata.</returns>
        public async Task<ModelMetadata> LoadAsync(ModelOptimization optimizations = default, CancellationToken cancellationToken = default)
        {
            try
            {
                if (_session is null)
                    return await CreateSession(optimizations, cancellationToken);

                if (HasOptimizationsChanged(optimizations))
                {
                    await UnloadAsync();
                    return await CreateSession(optimizations, cancellationToken);
                }
                return _metadata;
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("ErrorCode:RequirementNotRegistered"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Unloads the model session.
        /// </summary>
        /// <returns></returns>
        public async Task UnloadAsync()
        {
            // TODO: deadlock on model dispose when no synchronization context exists(console app)
            // Task.Yield seems to force a context switch resolving any issues, revist this
            await Task.Yield();

            if (_session is not null)
            {
                _session.Dispose();
                _metadata = null;
                _session = null;
            }
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when you do not have a known output shape.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public virtual IDisposableReadOnlyCollection<OrtValue> RunInference(ModelParameters parameters)
        {
            try
            {
                return _session.Run(parameters.RunOptions, parameters.InputNameValues, parameters.OutputNames);
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("Exiting due to terminate flag"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when the output shape is known
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public virtual async Task<IDisposableReadOnlyCollection<OrtValue>> RunInferenceAsync(ModelParameters parameters)
        {
            try
            {
                return new DisposableList<OrtValue>(await _session.RunAsync(parameters.RunOptions, parameters.InputNames, parameters.InputValues, parameters.OutputNames, parameters.OutputValues));
            }
            catch (OnnxRuntimeException ex)
            {
                if (ex.Message.Contains("Exiting due to terminate flag"))
                    throw new OperationCanceledException("Inference was canceled.", ex);
                throw;
            }
        }


        /// <summary>
        /// Creates the InferenceSession.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>The Sessions ModelMetadata.</returns>
        /// 
        private async Task<ModelMetadata> CreateSession(ModelOptimization optimizations, CancellationToken cancellationToken)
        {
            _options?.Dispose();
            _options = _configuration.ExecutionProvider.CreateSession(_configuration);
            cancellationToken.Register(_options.CancelSession, true);

            if (_configuration.IsOptimizationSupported)
                ApplyOptimizations(optimizations);

            _session = await Task.Run(() => new InferenceSession(_configuration.Path, _options), cancellationToken);
            _allocator = new OrtAllocator(_session, _configuration.ExecutionProvider.MemoryInfo);
            _metadata = new ModelMetadata(_session, _allocator);
            return _metadata;
        }


        /// <summary>
        /// Applies the optimizations.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        protected virtual void ApplyOptimizations(ModelOptimization optimizations)
        {
            _optimizations = optimizations;
            if (_optimizations != null)
            {
                _options.GraphOptimizationLevel = optimizations.OptimizationLevel.ToGraphOptimizationLevel();
                if (_optimizations.OptimizationLevel == Optimization.None)
                    return;

                foreach (var freeDimensionOverride in _optimizations.DimensionOverrides)
                {
                    if (freeDimensionOverride.Key.StartsWith("dummy_"))
                        continue;

                    _options.AddFreeDimensionOverrideByName(freeDimensionOverride.Key, freeDimensionOverride.Value);
                }
            }
        }


        /// <summary>
        /// Determines whether optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns><c>true</c> if changed; otherwise, <c>false</c>.</returns>
        public virtual bool HasOptimizationsChanged(ModelOptimization optimizations)
        {
            if (_optimizations == null && optimizations == null)
                return false; // No Optimizations set

            if (_optimizations == optimizations)
                return false; // Optimizations have not changed

            return true;
        }

        #region IDisposable

        private bool disposed = false;

        /// <summary>
        /// Disposes the managed and unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Disposes the managed and unmanaged resources.
        /// </summary>
        /// <param name="disposing">if set to <c>true</c> if disposing</param>
        protected virtual void Dispose(bool disposing)
        {
            if (disposed)
                return;

            if (disposing)
            {
                _allocator?.Dispose();
                _options?.Dispose();
                _session?.Dispose();
                _session = null;
            }

            disposed = true;
        }


        /// <summary>
        /// Finalizes an instance of the <see cref="ModelSession"/> class.
        /// </summary>
        ~ModelSession()
        {
            Dispose(false);
        }

        #endregion
    }


    /// <summary>
    /// Default ModelSession.
    /// Implements the <see cref="IDisposable" />
    /// </summary>
    /// <seealso cref="IDisposable" />
    public class ModelSession : ModelSession<ModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public ModelSession(ModelConfig configuration) : base(configuration) { }
    }
}
