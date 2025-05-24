// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;

namespace TensorStack.Core.Inference
{
    /// <summary>
    /// ModelSession class to manage lifetime of an InferenceSession ans its SessionOptions.
    /// </summary>
    /// <typeparam name="T">ModelConfig implementation</typeparam>
    /// <seealso cref="IDisposable" />
    public class ModelSession<T> : IDisposable where T : ModelConfig
    {
        private readonly T _configuration;
        private readonly bool _registerOrtExtensions;
        private readonly Func<SessionOptions> _sessionOptionsFactory;
        private ModelMetadata _metadata;
        private InferenceSession _session;
        private SessionOptions _sessionOptions;
        private ModelOptimization _optimizations;

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSession{T}"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <exception cref="System.IO.FileNotFoundException">Onnx model file not found, Path: {configuration.Path}</exception>
        private ModelSession(T configuration)
        {
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException($"Onnx model file not found, Path: {configuration.Path}", configuration.Path);

            _configuration = configuration;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public ModelSession(T configuration, bool useOrtExtensions = false)
            : this(configuration)
        {
            _registerOrtExtensions = useOrtExtensions;
            _sessionOptionsFactory = CreateDefaultSessionOptions;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ModelSession"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="sessionOptionsFactory">A factory to create session options.</param>
        public ModelSession(T configuration, Func<SessionOptions> sessionOptionsFactory)
            : this(configuration)
        {
            if (configuration.Provider != Provider.Custom)
                throw new ArgumentException("Provider must be set to Custom when using SessionOptionsFactory");

            _sessionOptionsFactory = sessionOptionsFactory;
        }

        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        public T Configuration => _configuration;

        /// <summary>
        /// Gets the InferenceSession.
        /// </summary>
        public InferenceSession Session => _session;

        /// <summary>
        /// Gets the SessionOptions.
        /// </summary>
        public SessionOptions SessionOptions => _sessionOptions;


        /// <summary>
        /// Loads the model session.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>ModelMetadata.</returns>
        public virtual ModelMetadata Load(ModelOptimization optimizations = default)
        {
            if (_session is null)
                return CreateSession(optimizations);

            if (HasOptimizationsChanged(optimizations))
            {
                Unload();
                return CreateSession(optimizations);
            }
            return _metadata;
        }


        /// <summary>
        /// Loads the model session asynchronously.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>ModelMetadata.</returns>
        public virtual async Task<ModelMetadata> LoadAsync(ModelOptimization optimizations = default, CancellationToken cancellationToken = default)
        {
            if (_session is null)
                return await CreateSessionAsync(optimizations, cancellationToken);

            if (HasOptimizationsChanged(optimizations))
            {
                await UnloadAsync();
                return await CreateSessionAsync(optimizations, cancellationToken);
            }
            return _metadata;
        }


        /// <summary>
        /// Unloads the model session.
        /// </summary>
        /// <returns></returns>
        public virtual void Unload()
        {
            _session?.Dispose();
            _metadata = null;
            _session = null;
        }


        /// <summary>
        /// Unloads the model session asynchronously.
        /// </summary>
        /// <returns>Task.</returns>
        public virtual Task UnloadAsync()
        {
            Unload();
            return Task.CompletedTask;
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when you do not have a known output shape.
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public virtual IDisposableReadOnlyCollection<OrtValue> RunInference(InferenceParameters parameters)
        {
            return _session.Run(parameters.RunOptions, parameters.InputNameValues, parameters.OutputNames);
        }


        /// <summary>
        /// Runs inference on the model with the suppied parameters, use this method when the output shape is known
        /// </summary>
        /// <param name="parameters">The parameters.</param>
        /// <returns></returns>
        public virtual async Task<IDisposableReadOnlyCollection<OrtValue>> RunInferenceAsync(InferenceParameters parameters)
        {
            return new DisposableList<OrtValue>(await _session.RunAsync(parameters.RunOptions, parameters.InputNames, parameters.InputValues, parameters.OutputNames, parameters.OutputValues));
        }


        /// <summary>
        /// Gets the default SessionOptions.
        /// </summary>
        /// <param name="useOrtExtensions">if set to <c>true</c> [use ort extensions].</param>
        /// <returns>SessionOptions.</returns>
        /// <exception cref="NotImplementedException"></exception>
        protected virtual SessionOptions CreateDefaultSessionOptions()
        {
            var sessionOptions = new SessionOptions();
            if (_registerOrtExtensions)
                sessionOptions.RegisterOrtExtensions();

            sessionOptions.GraphOptimizationLevel = GraphOptimizationLevel.ORT_DISABLE_ALL;
            switch (_configuration.Provider)
            {
                case Provider.CPU:
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;
                case Provider.DirectML:
                    sessionOptions.AppendExecutionProvider_DML(_configuration.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;
                case Provider.CUDA:
                    sessionOptions.AppendExecutionProvider_CUDA(_configuration.DeviceId);
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;
                case Provider.CoreML:
                    sessionOptions.AppendExecutionProvider_CoreML(CoreMLFlags.COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE);
                    sessionOptions.AppendExecutionProvider_CPU();
                    break;
                default:
                    throw new NotImplementedException();
            }
            return sessionOptions;
        }


        /// <summary>
        /// Creates the InferenceSession.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>The Sessions ModelMetadata.</returns>
        protected virtual ModelMetadata CreateSession(ModelOptimization optimizations)
        {
            _sessionOptions?.Dispose();
            _sessionOptions = _sessionOptionsFactory();

            return CreateSession(new InferenceSession(_configuration.Path, _sessionOptions), optimizations);
        }


        /// <summary>
        /// Creates the InferenceSession asynchronouly.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>The Sessions ModelMetadata.</returns>
        protected virtual async Task<ModelMetadata> CreateSessionAsync(ModelOptimization optimizations, CancellationToken cancellationToken = default)
        {
            _sessionOptions?.Dispose();
            _sessionOptions = _sessionOptionsFactory();

            var session = await Task.Run(() => new InferenceSession(_configuration.Path, _sessionOptions), cancellationToken);
            return CreateSession(session, optimizations);
        }


        /// <summary>
        /// Creates the InferenceSession.
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns>The Sessions ModelMetadata.</returns>
        protected virtual ModelMetadata CreateSession(InferenceSession newSession, ModelOptimization optimizations)
        {
            if (_configuration.IsOptimizationSupported)
                ApplyOptimizations(optimizations);

            _session?.Dispose();
            _session = newSession;
            _metadata = new ModelMetadata(_session);
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
                _sessionOptions.GraphOptimizationLevel = optimizations.OptimizationLevel.ToGraphOptimizationLevel();
                foreach (var freeDimensionOverride in _optimizations.DimensionOverrides)
                {
                    _sessionOptions.AddFreeDimensionOverrideByName(freeDimensionOverride.Key, freeDimensionOverride.Value);
                }
            }
        }


        /// <summary>
        /// Determines whether optimizations have changed
        /// </summary>
        /// <param name="optimizations">The optimizations.</param>
        /// <returns><c>true</c> if changed; otherwise, <c>false</c>.</returns>
        protected virtual bool HasOptimizationsChanged(ModelOptimization optimizations)
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
                _sessionOptions?.Dispose();
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
        /// <param name="useOrtExtensions">if set to <c>true</c> [use ort extensions].</param>
        public ModelSession(ModelConfig configuration, bool useOrtExtensions = false)
            : base(configuration, useOrtExtensions) { }
    }
}
