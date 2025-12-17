using Microsoft.Extensions.Logging;
using System;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TensorStack.Python.Config;
using TensorStack.Python.Options;

namespace TensorStack.Python
{
    public sealed class PythonServer : IDisposable
    {
        private readonly ILogger _logger;
        private readonly NamedPipeServerStream _objectPipe;
        private readonly NamedPipeServerStream _messagePipe;
        private readonly Channel<PythonProgress> _progressQueue;
        private readonly IProgress<PythonProgress> _progressCallback;
        private readonly PipelineConfig _pipelineConfig;
        private readonly ServerConfig _serverConfig;
        private readonly PythonService _pythonService;
        private readonly int _statusRefresh = 250;
        private CancellationTokenSource _cancellationTokenSource;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonServer"/> class.
        /// </summary>
        /// <param name="serverConfig">The server configuration.</param>
        /// <param name="pipelineConfig">The pipeline configuration.</param>
        /// <param name="logger">The logger.</param>
        public PythonServer(ServerConfig serverConfig, PipelineConfig pipelineConfig, ILogger logger = default)
        {
            _logger = logger;
            _serverConfig = serverConfig;
            _pipelineConfig = pipelineConfig;
            _objectPipe = new NamedPipeServerStream(ServerConfig.ObjectPipeName, PipeDirection.Out, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
            _messagePipe = new NamedPipeServerStream(ServerConfig.MessagePipeName, PipeDirection.InOut, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous);
            _progressQueue = Channel.CreateUnbounded<PythonProgress>();
            _progressCallback = new Progress<PythonProgress>(p => _progressQueue.Writer.TryWrite(p));
            _pythonService = new PythonService(_serverConfig, _progressCallback, logger);
        }


        /// <summary>
        /// Start the Server loop
        /// </summary>
        /// <param name="isRebuild">if set to <c>true</c> [is rebuild].</param>
        /// <param name="isReinstall">if set to <c>true</c> [is reinstall].</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task StartAsync(bool isRebuild, bool isReinstall, CancellationToken cancellationToken = default)
        {
            CallbackMessage("Starting Server...", "Initialize");
            _logger?.LogInformation($"[StartAsync] Waiting for connection");

            // Progress Loop
            await _objectPipe.WaitForConnectionAsync(cancellationToken);
            _ = ProcessProgressQueueAsync();

            _logger?.LogInformation($"[StartAsync] Client connected.");

            // Create Envrironment
            await _pythonService.CreateEnvironmentAsync(isRebuild, isReinstall);

            CallbackMessage("Loading Pipeline...", "Initialize");
            using (var pythonProxy = new PythonProxy(_pipelineConfig.Pipeline, _logger))
            {
                // Load Pipeline
                await pythonProxy
                     .LoadAsync(_pipelineConfig)
                     .WithPythonLogging(pythonProxy, _progressCallback, _statusRefresh);

                CallbackMessage("", "Initialize");
                await _messagePipe.WaitForConnectionAsync(cancellationToken);

                // Generate Loop
                _logger?.LogInformation($"[StartAsync] Start generate loop.");
                while (!cancellationToken.IsCancellationRequested)
                {
                    try
                    {
                        // Read Request
                        var message = await _messagePipe.ReceiveMessage<PythonRequestMessage>();
                        _logger?.LogInformation($"[StartAsync] Message received.");
                        if (message.IsStartRequest)
                        {
                            _logger?.LogInformation($"[StartAsync] Start Requested.");
                            await _messagePipe.SendResponse(cancellationToken);
                            continue;
                        }

                        if (message.IsStopRequest)
                        {
                            _logger?.LogInformation($"[StartAsync] Stop Requested.");
                            await _messagePipe.SendResponse(cancellationToken);
                            return;
                        }

                        // Generate Response
                        CallbackMessage("Generating...");
                        var response = await pythonProxy
                            .GenerateAsync(message.Options, cancellationToken)
                            .WithPythonLogging(pythonProxy, _progressCallback, _statusRefresh);

                        _logger?.LogInformation($"[StartAsync] Response generated.");

                        // Send Response
                        await _messagePipe.SendMessage(new PythonResponseMessage
                        {
                            Tensors = [response]
                        });
                        _logger?.LogInformation($"[StartAsync] Response sent.");
                        CallbackMessage("Generation Complete.");
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, "[StartAsync] An exception occurred");
                        throw;
                    }
                }
            }
        }


        /// <summary>
        /// Process the progress queue
        /// </summary>
        /// <param name="progressQueue">The progress queue.</param>
        private async Task ProcessProgressQueueAsync()
        {
            using (_cancellationTokenSource = new CancellationTokenSource())
            {
                await foreach (var progress in _progressQueue.Reader.ReadAllAsync(_cancellationTokenSource.Token))
                {
                    await _objectPipe.SendObject(progress);
                }
            }
        }


        /// <summary>
        /// Send a callback message.
        /// </summary>
        /// <param name="message">The message.</param>
        private void CallbackMessage(string message, string process = "Generate")
        {
            _progressCallback?.Report(new PythonProgress
            {
                Message = message,
                Process = process
            });
        }



        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _cancellationTokenSource?.SafeCancel();
            _objectPipe?.Dispose();
            _messagePipe?.Dispose();
        }
    }
}
