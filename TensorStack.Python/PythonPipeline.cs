using CSnakes.Runtime;
using CSnakes.Runtime.Python;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Python.Common;
using TensorStack.Python.Config;

namespace TensorStack.Python
{
    /// <summary>
    /// PipelineProxy: Proxy between Python and C#
    /// </summary>
    public sealed class PythonPipeline : IDisposable
    {
        private readonly ILogger _logger;
        private readonly string _pipelineName;
        private readonly PipelineConfig _configuration;
        private readonly int _progressRefresh;
        private readonly IProgress<PipelineProgress> _progressCallback;
        private readonly CancellationTokenSource _progressCancellation;
        private PyObject _module;
        private PyObject _functionLoad;
        private PyObject _functionUnload;
        private PyObject _functionCancel;
        private PyObject _functionGenerate;
        private PyObject _functionGetStepLatent;
        private PyObject _functionGetLogs;


        /// <summary>
        /// Initializes a new instance of the <see cref="PythonPipeline"/> class.
        /// </summary>
        /// <param name="moduleName">Name of the module.</param>
        /// <param name="logger">The logger.</param>
        public PythonPipeline(PipelineConfig configuration, IProgress<PipelineProgress> progressCallback = default, ILogger logger = default)
        {
            _logger = logger;
            _configuration = configuration;
            _progressRefresh = 250;
            _progressCallback = progressCallback;
            _progressCancellation = new CancellationTokenSource();
            _pipelineName = _configuration.Pipeline;
            using (GIL.Acquire())
            {
                _logger?.LogInformation("Importing module {ModuleName}", _pipelineName);
                _module = Import.ImportModule(_pipelineName);
                BindFunctions();
            }
            _ = LoggingLoop(_progressRefresh);
        }


        /// <summary>
        /// Reloads the module.
        /// </summary>
        public void ReloadModule()
        {
            using (GIL.Acquire())
            {
                _logger?.LogInformation("Reloading module {ModuleName}", _pipelineName);

                Import.ReloadModule(ref _module);
                UnbindFunctions();
                BindFunctions();
            }
        }


        /// <summary>
        /// Loads the proxy
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public Task<bool> LoadAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("Invoking Python function: {FunctionName}", "load");

                        var loraConfig = _configuration.LoraAdapters?.Select(x => (x.Path, x.Weights, x.Name));
                        using (var modelName = PyObject.From(_configuration.Path))
                        using (var processType = PyObject.From(_configuration.ProcessType.ToString()))
                        using (var isModelOffloadEnabled = PyObject.From(_configuration.IsModelOffloadEnabled))
                        using (var isFullOffloadEnabled = PyObject.From(_configuration.IsFullOffloadEnabled))
                        using (var isVaeSlicingEnabled = PyObject.From(_configuration.IsVaeSlicingEnabled))
                        using (var isVaeTilingEnabled = PyObject.From(_configuration.IsVaeTilingEnabled))
                        using (var device = PyObject.From(_configuration.Device))
                        using (var deviceId = PyObject.From(_configuration.DeviceId))
                        using (var dataType = PyObject.From(_configuration.DataType.ToString().ToLower()))
                        using (var variant = PyObject.From(_configuration.Variant))
                        using (var cacheDir = PyObject.From(_configuration.CacheDirectory))
                        using (var secureToken = PyObject.From(_configuration.SecureToken))
                        using (var loraAdapters = PyObject.From(loraConfig))
                        using (var pythonResult = _functionLoad.Call(modelName, processType, device, deviceId, dataType, variant, cacheDir, secureToken, isModelOffloadEnabled, isFullOffloadEnabled, isVaeSlicingEnabled, isVaeTilingEnabled, loraAdapters))
                        {
                            return pythonResult.BareImportAs<bool, PyObjectImporters.Boolean>();
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Unload the proxy
        /// </summary>
        public Task<bool> UnloadAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("Invoking Python function: {FunctionName}", "unload");

                        using (var pythonResult = _functionUnload.Call())
                        {
                            return pythonResult.BareImportAs<bool, PyObjectImporters.Boolean>();
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Generate
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task<Tensor<float>> GenerateAsync(PipelineOptions options, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("Invoking Python function: {FunctionName}", "generate");

                        cancellationToken.Register(() => GenerateCancelAsync(), true);

                        var imageInput = options.ImageInput?.GetChannels(3).ToTensor();
                        var loraConfig = options.LoraOptions?.ToDictionary(k => k.Name, v => v.Strength);
                        using (var prompt = PyObject.From(options.Prompt))
                        using (var negativePrompt = PyObject.From(options.NegativePrompt))
                        using (var guidance = PyObject.From(options.GuidanceScale))
                        using (var guidance2 = PyObject.From(options.GuidanceScale2))
                        using (var steps = PyObject.From(options.Steps))
                        using (var steps2 = PyObject.From(options.Steps2))
                        using (var height = PyObject.From(options.Height))
                        using (var width = PyObject.From(options.Width))
                        using (var seed = PyObject.From(options.Seed))
                        using (var scheduler = PyObject.From(options.Scheduler.ToString()))
                        using (var numFrames = PyObject.From(options.Frames))
                        using (var shift = PyObject.From(options.Shift))
                        using (var flowShift = PyObject.From(options.FlowShift))
                        using (var strength = PyObject.From(options.Strength))
                        using (var loraOptions = PyObject.From(loraConfig))
                        using (var inputData = PyObject.From(imageInput?.Memory.ToArray()))
                        using (var inputShape = PyObject.From(imageInput.Dimensions.ToArray()))
                        using (var pythonResult = _functionGenerate.Call(prompt, negativePrompt, guidance, guidance2, steps, steps2, height, width, seed, scheduler, numFrames, shift, flowShift, strength, loraOptions, inputData, inputShape))
                        {
                            var result = pythonResult
                                 .BareImportAs<IPyBuffer, PyObjectImporters.Buffer>()
                                 .ToTensor()
                                 .Normalize(Normalization.OneToOne);
                            return result;
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Gets the logs.
        /// </summary>
        public Task<IReadOnlyList<string>> GetLogsAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        using (var pythonResult = _functionGetLogs.Call())
                        {
                            return pythonResult.BareImportAs<IReadOnlyList<string>, PyObjectImporters.List<string, PyObjectImporters.String>>();
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Gets the step latents.
        /// </summary>
        public Task<Tensor<float>> GetStepLatentAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("Invoking Python function: {FunctionName}", "get_step_latent");

                        using (var pythonResult = _functionGetStepLatent.Call())
                        {
                            return pythonResult
                                .BareImportAs<IPyBuffer, PyObjectImporters.Buffer>()
                                .ToTensor();
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Cancel Generation
        /// </summary>
        /// <returns>Task.</returns>
        private Task GenerateCancelAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("Invoking Python function: {FunctionName}", "cancel");

                        using (var pythonResult = _functionCancel.Call())
                        {
                            return;
                        }
                    }
                    catch (PythonInvocationException ex)
                    {
                        throw HandlePythonException(ex);
                    }
                }
            });
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _logger?.LogInformation("Disposing module {ModuleName}", _pipelineName);
            _progressCancellation.SafeCancel();
            _progressCancellation.Dispose();
            UnbindFunctions();
            _module.Dispose();
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Binds the functions.
        /// </summary>
        private void BindFunctions()
        {
            _functionLoad = _module.GetAttr("load");
            _functionUnload = _module.GetAttr("unload");
            _functionCancel = _module.GetAttr("generateCancel");
            _functionGenerate = _module.GetAttr("generate");
            _functionGetStepLatent = _module.GetAttr("getStepLatent");
            _functionGetLogs = _module.GetAttr("getLogs");
        }


        /// <summary>
        /// Unbinds the functions.
        /// </summary>
        private void UnbindFunctions()
        {
            _functionLoad.Dispose();
            _functionUnload.Dispose();
            _functionCancel.Dispose();
            _functionGenerate.Dispose();
            _functionGetStepLatent.Dispose();
            _functionGetLogs.Dispose();
        }


        /// <summary>
        /// Logging loop.
        /// </summary>
        /// <param name="refreshRate">The refresh rate.</param>
        private async Task LoggingLoop(int refreshRate)
        {
            while (!_progressCancellation.IsCancellationRequested)
            {
                var logs = await GetLogsAsync();
                foreach (var progress in LogParser.ParseLogs(logs))
                {
                    _logger?.LogInformation("[PythonRuntime] {message}", progress.Message);
                    _progressCallback?.Report(progress);
                }
                await Task.Delay(refreshRate, _progressCancellation.Token);
            }
        }


        /// <summary>
        /// Handles the python exception.
        /// </summary>
        /// <param name="ex">The ex.</param>
        /// <returns>Exception.</returns>
        private Exception HandlePythonException(PythonInvocationException ex)
        {
            if (ex.InnerException is PythonRuntimeException pyex)
            {
                _logger?.LogError(pyex, "{PythonExceptionType} exception occured", ex.PythonExceptionType);
                if (!pyex.PythonStackTrace.IsNullOrEmpty())
                    _logger?.LogError(string.Join(Environment.NewLine, pyex.PythonStackTrace));

                return new Exception(pyex.Message, pyex);
            }

            _logger?.LogError(ex, "{PythonExceptionType} exception occured", ex.PythonExceptionType);
            return new Exception(ex.Message, ex);
        }
    }
}

