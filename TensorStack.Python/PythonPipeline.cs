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
        private readonly IProgress<PipelineProgress> _progressCallback;
        private PyObject _module;
        private PyObject _functionLoad;
        private PyObject _functionReload;
        private PyObject _functionUnload;
        private PyObject _functionDownload;
        private PyObject _functionCancel;
        private PyObject _functionGenerate;
        private PyObject _functionGetLogs;
        private PyObject _functionGetNotifications;
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
            _progressCallback = progressCallback;
            _pipelineName = _configuration.Pipeline;
            using (GIL.Acquire())
            {
                _logger?.LogInformation("[PythonPipeline] [Load] Importing pipeline module '{pipelineName}'.", _pipelineName);
                _module = Import.ImportModule(_pipelineName);
                BindFunctions();
            }
            _ = LoggingLoop(50);
            _ = NotificationLoop(25);
        }


        /// <summary>
        /// Reloads the module.
        /// </summary>
        public void ReloadModule()
        {
            using (GIL.Acquire())
            {
                _logger?.LogInformation("[PythonPipeline] [ReloadModule] Reloading module.");

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
                        _logger?.LogInformation("[PythonPipeline] [Load] Loading pipeline.");

                        var pipelineConfigDict = _configuration.ToPythonDictionary();
                        using (var pipelineConfig = PyObject.From(pipelineConfigDict))
                        using (var pythonResult = _functionLoad.Call(pipelineConfig))
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
        /// Reloads the pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>Task&lt;System.Boolean&gt;.</returns>
        public Task<bool> ReloadAsync(PipelineReloadOptions options)
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("[PythonPipeline] [Reload] Reloading pipeline.");

                        var configuration = _configuration with
                        {
                            ProcessType = options.ProcessType,
                            ControlNet = options.ControlNet,
                            LoraAdapters = options.LoraAdapters,
                        };

                        var pipelineConfigDict = configuration.ToPythonDictionary();
                        using (var pipelineConfig = PyObject.From(pipelineConfigDict))
                        using (var pythonResult = _functionReload.Call(pipelineConfig))
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
                        _logger?.LogInformation("[PythonPipeline] [Unload] Unloading pipeline.");

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
        /// Download pipeline components
        /// </summary>
        public Task<bool> DownloadAsync()
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("[PythonPipeline] [Download] Downloading pipeline.");

                        var pipelineConfigDict = _configuration.ToPythonDictionary();
                        using (var pipelineConfig = PyObject.From(pipelineConfigDict))
                        using (var pythonResult = _functionDownload.Call(pipelineConfig))
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
        public Task<IReadOnlyList<Tensor<float>>> GenerateAsync(PipelineOptions options, CancellationToken cancellationToken = default)
        {
            return Task.Run<IReadOnlyList<Tensor<float>>>(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        _logger?.LogInformation("[PythonPipeline] [Generate] Executing pipeline.");
                        cancellationToken.Register(() => GenerateCancelAsync(), true);

                        var inputTensors = GetInputData(options);
                        var controlInputTensors = GetControlInputData(options);
                        var inferenceOptionsDict = options.ToPythonDictionary();
                        using (var inferenceOptions = PyObject.From(inferenceOptionsDict))
                        using (var imageData = PyObject.From(inputTensors))
                        using (var controlNetData = PyObject.From(controlInputTensors))
                        using (var pythonResults = _functionGenerate.Call(inferenceOptions, imageData, controlNetData))
                        {
                            return pythonResults.AsBareEnumerable<IPyBuffer, PyObjectImporters.Buffer>()
                                .Select(x => x.ToTensor().Normalize(Normalization.OneToOne))
                                .ToList();
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
        /// Gets the Notifications.
        /// </summary>
        public Task<IReadOnlyList<PipelineProgress>> GetNotificationsAsync()
        {
            return Task.Run<IReadOnlyList<PipelineProgress>>(() =>
            {
                using (GIL.Acquire())
                {
                    try
                    {
                        using (var pythonResult = _functionGetNotifications.Call())
                        {
                            var pythonResults = pythonResult.BareImportAs<
                                IReadOnlyList<(string, IPyBuffer)>,
                                PyObjectImporters.List<(string, IPyBuffer),
                                PyObjectImporters.Tuple<string, IPyBuffer, PyObjectImporters.String, PyObjectImporters.Buffer>>>();

                            return pythonResults
                                .Select(x => PipelineProgress.Create(x.Item1, x.Item2.ToTensor()))
                                .Where(x => x?.Key != null)
                                .ToList();
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
                        _logger?.LogInformation("[PythonPipeline] [GenerateCancel] Canceling generation.");

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
            _logger?.LogInformation("[PythonPipeline] [Dispose] Disposing pipeline.");
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
            _functionReload = _module.GetAttr("reload");
            _functionUnload = _module.GetAttr("unload");
            _functionDownload = _module.GetAttr("download");
            _functionCancel = _module.GetAttr("generateCancel");
            _functionGenerate = _module.GetAttr("generate");
            _functionGetLogs = _module.GetAttr("getLogs");
            _functionGetNotifications = _module.GetAttr("getNotifications");
        }


        /// <summary>
        /// Unbinds the functions.
        /// </summary>
        private void UnbindFunctions()
        {
            _functionLoad.Dispose();
            _functionReload.Dispose();
            _functionUnload.Dispose();
            _functionDownload.Dispose();
            _functionCancel.Dispose();
            _functionGenerate.Dispose();
            _functionGetLogs.Dispose();
            _functionGetNotifications.Dispose();
        }


        /// <summary>
        /// Logging loop.
        /// </summary>
        /// <param name="refreshRate">The refresh rate.</param>
        private async Task LoggingLoop(int refreshRate)
        {
            while (_isRunning)
            {
                var logEntries = await GetLogsAsync();
                foreach (var logEntry in LogParser.ParseLogs(logEntries).OrderBy(x => x.Timestamp))
                {
                    if (string.IsNullOrWhiteSpace(logEntry?.Message))
                        continue;

                    _logger?.LogInformation("[PythonPipeline] [PythonRuntime] [{Timestamp}] {Message}", logEntry.Timestamp.ToString("hh:mm:ss:fff"), logEntry.Message);
                }
                await Task.Delay(refreshRate);
            }
        }


        /// <summary>
        /// Notification loop.
        /// </summary>
        /// <param name="refreshRate">The refresh rate.</param>
        private async Task NotificationLoop(int refreshRate)
        {
            while (_isRunning)
            {
                var progressItems = await GetNotificationsAsync();
                if (!progressItems.IsNullOrEmpty())
                {
                    foreach (var progress in progressItems)
                    {
                        _progressCallback?.Report(progress);
                        _logger?.LogDebug("[PythonPipeline] [PythonRuntime] {Progress}", progress);
                    }
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

                _logger?.LogError(pyex, "[PythonPipeline] [PythonRuntime] {PythonExceptionType} exception occurred", ex.PythonExceptionType);
                if (!pyex.PythonStackTrace.IsNullOrEmpty())
                    _logger?.LogError(string.Join(Environment.NewLine, pyex.PythonStackTrace));

                return new Exception(pyex.Message, pyex);
            }

            _logger?.LogError(ex, "[PythonPipeline] [PythonRuntime] {PythonExceptionType} exception occurred", ex.PythonExceptionType);
            return new Exception(ex.Message, ex);
        }


        private List<(float[], int[])> GetInputData(PipelineOptions options)
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


        private List<(float[], int[])> GetControlInputData(PipelineOptions options)
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