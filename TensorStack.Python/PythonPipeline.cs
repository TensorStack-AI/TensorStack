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
        private PyObject _module;
        private PyObject _functionLoad;
        private PyObject _functionUnload;
        private PyObject _functionCancel;
        private PyObject _functionGenerate;
        private PyObject _functionGetStepLatent;
        private PyObject _functionGetLogs;
        private bool _isRunning;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonPipeline"/> class.
        /// </summary>
        /// <param name="moduleName">Name of the module.</param>
        /// <param name="logger">The logger.</param>
        public PythonPipeline(PipelineConfig configuration, IProgress<PipelineProgress> progressCallback = default, ILogger logger = default)
        {
            _logger = logger;
            _isRunning = true;
            _configuration = configuration;
            _progressRefresh = 250;
            _progressCallback = progressCallback;
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

                        var pipelineConfigDict = _configuration.ToPythonDictionary("lora_adapters");
                        var loraConfigDict = _configuration.LoraAdapters?.Select(x => (x.Path, x.Weights, x.Name));

                        using (var pipelineConfig = PyObject.From(pipelineConfigDict))
                        using (var loraConfig = PyObject.From(loraConfigDict))
                        using (var pythonResult = _functionLoad.Call(pipelineConfig, loraConfig))
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

                        var images = GetImageData(options);
                        var controlNetImages = GetControlImageData(options);

                        var schedulerOptionsDict = options.SchedulerOptions.ToPythonDictionary();
                        var inferenceOptionsDict = options.ToPythonDictionary("scheduler_options", "lora_options");
                        var loraOptionsDict = options.LoraOptions?.ToDictionary(k => k.Name, v => v.Strength);

                        using (var inferenceOptions = PyObject.From(inferenceOptionsDict))
                        using (var schedulerOptions = PyObject.From(schedulerOptionsDict))
                        using (var loraOptions = PyObject.From(loraOptionsDict))
                        using (var imageData = PyObject.From(images))
                        using (var controlNetData = PyObject.From(controlNetImages))
                        using (var pythonResult = _functionGenerate.Call(inferenceOptions, schedulerOptions, loraOptions, imageData, controlNetData))
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
            _isRunning = false;
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
            while (_isRunning)
            {
                var logs = await GetLogsAsync();
                foreach (var progress in LogParser.ParseLogs(logs))
                {
                    _logger?.LogInformation("[PythonRuntime] {Message}", progress.Message);
                    _progressCallback?.Report(progress);
                }
                await Task.Delay(refreshRate);
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
                if (ex.InnerException.Message.Equals("Operation Canceled"))
                    return new OperationCanceledException();

                _logger?.LogError(pyex, "{PythonExceptionType} exception occurred", ex.PythonExceptionType);
                if (!pyex.PythonStackTrace.IsNullOrEmpty())
                    _logger?.LogError(string.Join(Environment.NewLine, pyex.PythonStackTrace));

                return new Exception(pyex.Message, pyex);
            }

            _logger?.LogError(ex, "{PythonExceptionType} exception occurred", ex.PythonExceptionType);
            return new Exception(ex.Message, ex);
        }


        private List<(float[], int[])> GetImageData(PipelineOptions options)
        {
            if (options.InputImages.IsNullOrEmpty())
                return null;

            var inputData = new List<(float[], int[])>();
            foreach (var imageInput in options.InputImages)
            {
                var imageTensor = imageInput.GetChannels(3);
                inputData.Add((imageTensor.Span.ToArray(), imageTensor.Dimensions.ToArray()));
            }
            return inputData;
        }


        private List<(float[], int[])> GetControlImageData(PipelineOptions options)
        {
            if (options.InputControlImages.IsNullOrEmpty())
                return null;

            var inputData = new List<(float[], int[])>();
            foreach (var imageInput in options.InputControlImages)
            {
                var imageTensor = imageInput.GetChannels(3);
                inputData.Add((imageTensor.Span.ToArray(), imageTensor.Dimensions.ToArray()));
            }
            return inputData;
        }
    }
}

