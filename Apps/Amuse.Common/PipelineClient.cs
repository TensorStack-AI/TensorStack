using Amuse.Common.Config;
using Amuse.Common.Message;
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace Amuse.Common
{
    public class PipelineClient : IDisposable
    {
        private readonly ILogger _logger;
        private readonly ClientConfig _config;
        private readonly ServerConfig _serverConfig;
        private readonly NamedPipeClientStream _commandChannel;
        private readonly NamedPipeClientStream _pipelineChannel;
        private readonly NamedPipeClientStream _progressChannel;
        private readonly ProcessHandler _processHandler;
        private readonly IProgress<PipelineProgress> _progressCallback;

        private CancellationTokenSource _progressCancellation;
        private Process _serverProcess;
        private bool _isCanceled;

        protected PipelineClient(ClientConfig config, ServerConfig serverConfig, IProgress<PipelineProgress> progressCallback, ILogger logger = default)
        {
            _logger = logger;
            _config = config;
            _serverConfig = serverConfig;
            _progressCallback = progressCallback;
            _processHandler = new ProcessHandler();
            _progressCancellation = new CancellationTokenSource();
            _commandChannel = new NamedPipeClientStream(".", _serverConfig.ChannelCommand, PipeDirection.InOut, PipeOptions.Asynchronous);
            _pipelineChannel = new NamedPipeClientStream(".", _serverConfig.ChannelPipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
            _progressChannel = new NamedPipeClientStream(".", _serverConfig.ChannelProgress, PipeDirection.In, PipeOptions.Asynchronous);
            _ = ProcessProgressQueueAsync(_progressCallback);
        }

        public PipelineClient(ClientConfig config, IProgress<PipelineProgress> progressCallback, ILogger logger = default)
            : this(config, ServerConfig.GetConfig(config.ServerType), progressCallback, logger) { }


        /// <summary>
        /// Start client as an asynchronous operation.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task StartAsync(CancellationToken cancellationToken)
        {
            _isCanceled = false;

            // Start Server
            await StartServerAsync();

            try
            {
                // Connect Pipes
                await Task.WhenAll
                (
                    _commandChannel.ConnectAsync(cancellationToken),
                    _progressChannel.ConnectAsync(cancellationToken),
                    _pipelineChannel.ConnectAsync(cancellationToken)
                );

                // Start Environment
                await SendPipelineRequestAsync(new PipelineRequest(RequestType.Start), cancellationToken);
            }
            catch (OperationCanceledException)
            {
                await KillServerAsync();
                throw;
            }
        }


        /// <summary>
        /// Stop client
        /// </summary>
        public async Task StopAsync()
        {
            _isCanceled = true;
            await SendPipelineRequestAsync(new PipelineRequest(RequestType.Stop));
            await StopServerAsync(_serverProcess);
        }


        /// <summary>
        /// Kill server.
        /// </summary>
        public virtual async Task KillServerAsync()
        {
            if (_serverProcess is not null)
            {
                _serverProcess.Kill(true);
                await _serverProcess.WaitForExitAsync();
            }
        }


        /// <summary>
        /// Send a Pipeline request to the Server
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected virtual async Task<PipelineResponse> SendPipelineRequestAsync(PipelineRequest request, CancellationToken cancellationToken = default)
        {
            await _pipelineChannel.SendMessage(request, cancellationToken);
            var response = await _pipelineChannel.ReceiveMessage<PipelineResponse>(cancellationToken);
            if (response.IsError)
            {
                if (response.IsCanceled)
                    throw new OperationCanceledException(response.Error);

                throw new Exception(response.Error);
            }
            return response;
        }


        /// <summary>
        /// Send a Object request to the Server
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected virtual async Task<CommandResponse> SendObjectRequestAsync(CommandRequest request, CancellationToken cancellationToken = default)
        {
            await _commandChannel.SendObject(request, cancellationToken);
            var response = await _commandChannel.ReceiveObject<CommandResponse>(cancellationToken);
            if (response.IsError)
            {
                if (response.IsCanceled)
                    throw new OperationCanceledException(response.Error);

                throw new Exception(response.Error);
            }
            return response;
        }


        /// <summary>
        /// Start server
        /// </summary>
        protected virtual Task StartServerAsync()
        {
            var processInfo = new ProcessStartInfo
            {
                CreateNoWindow = !_config.IsDebugMode,
                UseShellExecute = false,
                FileName = Path.Combine(_config.ServerPath, _serverConfig.Executable),
                Arguments = _serverConfig.Arguments.IsNullOrEmpty() ? null : string.Join(' ', _serverConfig.Arguments)
            };

            // Environment Variables
            if (!_config.ServerVariables.IsNullOrEmpty())
            {
                foreach (var variable in _config.ServerVariables)
                    processInfo.Environment.Add(variable);
            }

            _serverProcess = Process.Start(processInfo);
            _processHandler.AddProcess(_serverProcess);
            return Task.CompletedTask;
        }


        /// <summary>
        /// Stop server
        /// </summary>
        /// <param name="serverProcess">The server process.</param>
        /// <param name="timeout">The timeout.</param>
        protected virtual async Task StopServerAsync(Process serverProcess, int timeout = 5000)
        {
            using (serverProcess)
            {
                var timeoutDelay = Task.Delay(timeout);
                await Task.WhenAny(timeoutDelay, serverProcess.WaitForExitAsync());
                if (!serverProcess.HasExited)
                {
                    serverProcess.Kill(true);
                    await serverProcess.WaitForExitAsync();
                }
            }
        }


        /// <summary>
        /// Process the progress queue
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        protected virtual async Task ProcessProgressQueueAsync(IProgress<PipelineProgress> progressCallback)
        {
            while (!_progressCancellation.IsCancellationRequested)
            {
                try
                {
                    if (!_progressChannel.IsConnected)
                    {
                        await Task.Delay(100);
                        continue;
                    }

                    var progress = await _progressChannel.ReceiveObject<PipelineProgress>(_progressCancellation.Token);
                    if (progress == null || _isCanceled)
                        continue;

                    progressCallback?.Report(progress);
                }
                catch (OperationCanceledException) { }
                catch (Exception)
                {
                    // _logger?.LogError(ex, $"[PipelineClient] [ProcessProgressQueueAsync] - An exception occurred processing progress");
                }
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public virtual void Dispose()
        {
            _progressCancellation?.SafeCancel();
            _progressCancellation?.Dispose();
            _progressChannel?.Dispose();
            _commandChannel?.Dispose();
            _pipelineChannel?.Dispose();
            _serverProcess?.Dispose();
        }





        /// <summary>
        /// Create the pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task CreateAsync(PipelineCreateOptions options, CancellationToken cancellationToken = default)
        {
            try
            {
                _isCanceled = false;
                await StartAsync(cancellationToken);
                await SendPipelineRequestAsync(new PipelineRequest(options), cancellationToken);
            }
            catch (OperationCanceledException)
            {
                await KillServerAsync();
                throw;
            }
            catch (Exception)
            {
                await KillServerAsync();
                throw;
            }
        }


        /// <summary>
        /// Load the pipeline
        /// </summary>
        /// <param name="pipeline">The pipeline.</param>
        /// <param name="mode">The mode.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task LoadAsync(PipelineLoadOptions pipeline, CancellationToken cancellationToken = default)
        {
            try
            {
                _isCanceled = false;
                await SendPipelineRequestAsync(new PipelineRequest(pipeline, RequestType.Load), cancellationToken);
            }
            catch (OperationCanceledException)
            {
                await KillServerAsync();
                throw;
            }
        }


        /// <summary>
        /// Reload the pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task ReloadAsync(PipelineReloadOptions options, CancellationToken cancellationToken = default)
        {
            try
            {
                _isCanceled = false;
                await SendPipelineRequestAsync(new PipelineRequest(options), cancellationToken);
            }
            catch (OperationCanceledException)
            {
                await KillServerAsync();
                throw;
            }
        }


        /// <summary>
        /// Unload the pipeline
        /// </summary>
        public async Task UnloadAsync()
        {
            _isCanceled = true;
            await SendPipelineRequestAsync(new PipelineRequest(RequestType.Unload));
            await StopAsync();
        }


        /// <summary>
        /// Cancel pipeline Load/Run
        /// </summary>
        public async Task CancelAsync()
        {
            _isCanceled = true;
            await SendObjectRequestAsync(new CommandRequest());
        }


        /// <summary>
        /// Run the pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        public async Task<Tensor<float>> RunAsync(PipelineRunOptions options)
        {
            _isCanceled = false;
            var response = await SendPipelineRequestAsync(new PipelineRequest(options));
            return response.Tensors.FirstOrDefault();
        }


        public async Task<PipelineTextResult[]> GenerateText(PipelineRunOptions options)
        {
            _isCanceled = false;
            var response = await SendPipelineRequestAsync(new PipelineRequest(options));
            return response.TextResponse.Results;
        }
    }
}
