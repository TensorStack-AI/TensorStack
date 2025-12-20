using CSnakes.Runtime.Python;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Python.Config;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    /// <summary>
    /// PythonProxy = Proxy between Python and C#
    /// </summary>
    public sealed class PythonProxy : IDisposable
    {
        private readonly ILogger _logger;
        private readonly string _moduleName;
        private PyObject _module;
        private PyObject _functionLoad;
        private PyObject _functionUnload;
        private PyObject _functionCancel;
        private PyObject _functionGenerate;
        private PyObject _functionGetStepLatent;
        private PyObject _functionGetLogs;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonProxy"/> class.
        /// </summary>
        /// <param name="moduleName">Name of the module.</param>
        /// <param name="logger">The logger.</param>
        public PythonProxy(string moduleName, ILogger logger = default)
        {
            _logger = logger;
            _moduleName = moduleName;
            using (GIL.Acquire())
            {
                _logger?.LogDebug("Importing module {ModuleName}", _moduleName);
                _module = Import.ImportModule(_moduleName);
                BindFunctions();
            }
        }

        public string Name => _moduleName;
        public ILogger Logger => _logger;


        /// <summary>
        /// Reloads the module.
        /// </summary>
        public void ReloadModule()
        {
            _logger?.LogDebug("Reloading module {ModuleName}", _moduleName);

            using (GIL.Acquire())
            {
                Import.ReloadModule(ref _module);
                UnbindFunctions();
                BindFunctions();
            }
        }

        /// <summary>
        /// Loads the proxy
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public Task<bool> LoadAsync(PipelineConfig configuration)
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    _logger?.LogDebug("Invoking Python function: {FunctionName}", "load");

                    var loraConfig = configuration.LoraAdapters?.Select(x => (x.Path, x.Weights, x.Name));

                    using (var modelName = PyObject.From(configuration.Path))
                    using (var processType = PyObject.From(configuration.ProcessType.ToString()))
                    using (var isModelOffloadEnabled = PyObject.From(configuration.IsModelOffloadEnabled))
                    using (var isFullOffloadEnabled = PyObject.From(configuration.IsFullOffloadEnabled))
                    using (var isVaeSlicingEnabled = PyObject.From(configuration.IsVaeSlicingEnabled))
                    using (var isVaeTilingEnabled = PyObject.From(configuration.IsVaeTilingEnabled))
                    using (var device = PyObject.From(configuration.Device))
                    using (var deviceId = PyObject.From(configuration.DeviceId))
                    using (var dataType = PyObject.From(configuration.DataType.ToString().ToLower()))
                    using (var variant = PyObject.From(configuration.Variant))
                    using (var cacheDir = PyObject.From(configuration.CacheDirectory))
                    using (var secureToken = PyObject.From(configuration.SecureToken))
                    using (var loraAdapters = PyObject.From(loraConfig))
                    using (var pythonResult = _functionLoad.Call(modelName, processType, device, deviceId, dataType, variant, cacheDir, secureToken, isModelOffloadEnabled, isFullOffloadEnabled, isVaeSlicingEnabled, isVaeTilingEnabled, loraAdapters))
                    {
                        return pythonResult.BareImportAs<bool, PyObjectImporters.Boolean>();
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
                    _logger?.LogDebug("Invoking Python function: {FunctionName}", "unload");

                    using (var pythonResult = _functionUnload.Call())
                    {
                        return pythonResult.BareImportAs<bool, PyObjectImporters.Boolean>();
                    }
                }
            });
        }


        /// <summary>
        /// Generate
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public Task<Tensor<float>> GenerateAsync(PythonOptions options, List<Tensor<float>> inputTensors, CancellationToken cancellationToken = default)
        {
            return Task.Run(() =>
            {
                using (GIL.Acquire())
                {
                    _logger?.LogDebug("Invoking Python function: {FunctionName}", "generate");
                    cancellationToken.Register(() => GenerateCancelAsync(), true);

                    var inputTensor = inputTensors?.FirstOrDefault();
                    var loraConfig = options.LoraOptions?.ToDictionary(k => k.Name, v => v.Strength);
                    using (var prompt = PyObject.From(options.Prompt))
                    using (var negativePrompt = PyObject.From(options.NegativePrompt))
                    using (var guidanceScale = PyObject.From(options.GuidanceScale))
                    using (var steps = PyObject.From(options.Steps))
                    using (var height = PyObject.From(options.Height))
                    using (var width = PyObject.From(options.Width))
                    using (var seed = PyObject.From(options.Seed))
                    using (var scheduler = PyObject.From(options.Scheduler.ToString()))
                    using (var numFrames = PyObject.From(options.Frames))
                    using (var shift = PyObject.From(options.Shift))
                    using (var flowShift = PyObject.From(options.FlowShift))
                    using (var strength = PyObject.From(options.Strength))
                    using (var loraOptions = PyObject.From(loraConfig))
                    using (var inputData = PyObject.From(inputTensor?.Memory.ToArray()))
                    using (var inputShape = PyObject.From(inputTensor?.Dimensions.ToArray()))
                    using (var pythonResult = _functionGenerate.Call(prompt, negativePrompt, guidanceScale, steps, height, width, seed, scheduler, numFrames, shift, flowShift, strength, loraOptions, inputData, inputShape))
                    {
                        var result = pythonResult
                             .BareImportAs<IPyBuffer, PyObjectImporters.Buffer>()
                             .ToTensor()
                             .Normalize(Normalization.OneToOne);
                        return result;
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
                    //_logger?.LogDebug("Invoking Python function: {FunctionName}", "get_pipeline_status");

                    using (var pythonResult = _functionGetLogs.Call())
                    {
                        return pythonResult.BareImportAs<IReadOnlyList<string>, PyObjectImporters.List<string, PyObjectImporters.String>>();
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
                    _logger?.LogDebug("Invoking Python function: {FunctionName}", "get_step_latent");

                    using (var pythonResult = _functionGetStepLatent.Call())
                    {
                        return pythonResult
                            .BareImportAs<IPyBuffer, PyObjectImporters.Buffer>()
                            .ToTensor();
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
                    _logger?.LogDebug("Invoking Python function: {FunctionName}", "cancel");

                    using (var pythonResult = _functionCancel.Call())
                    {
                        return;
                    }
                }
            });
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _logger?.LogDebug("Disposing module {ModuleName}", _moduleName);

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
    }

}