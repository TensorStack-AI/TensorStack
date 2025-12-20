using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TensorStack.Python.Config;
using TensorStack.Python.Common;

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
        private readonly int _statusRefresh = 500;
        private CancellationTokenSource _cancellationTokenSource;
        private bool _isRunning;

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
            _logger?.LogInformation($"[PythonServer][StartAsync] Waiting for connection");

            // Progress Loop
            await _objectPipe.WaitForConnectionAsync(cancellationToken);
            _ = ProcessProgressQueueAsync();

            _logger?.LogInformation($"[PythonServer] [StartAsync] Client connected.");

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
                _logger?.LogInformation($"[PythonServer] [StartAsync] Start generate loop.");
                while (!cancellationToken.IsCancellationRequested)
                {
                    try
                    {
                        // Read Request
                        var message = await _messagePipe.ReceiveMessage<PythonRequestMessage>();
                        _logger?.LogInformation($"[PythonServer] [StartAsync] {message.Type} message received.");

                        // Start Server
                        if (message.Type == PythonMessageType.Start)
                        {
                            _isRunning = true;
                            await _messagePipe.SendResponse(cancellationToken);
                            _logger?.LogInformation($"[PythonServer] [StartAsync] Server started.");
                            continue;
                        }

                        // Stop Server
                        if (message.Type == PythonMessageType.Stop)
                        {
                            _isRunning = false;
                            _cancellationTokenSource?.SafeCancel();
                            await _messagePipe.SendResponse(cancellationToken);
                            return;
                        }

                        if (!_isRunning)
                        {
                            _logger?.LogError($"[PythonServer] [StartAsync] Server not started...");
                            continue;
                        }

                        // Generate Response
                        CallbackMessage("Generating...");
                        var response = await pythonProxy
                            .GenerateAsync(message.Options, message.Tensors, cancellationToken)
                            .WithPythonLogging(pythonProxy, _progressCallback, _statusRefresh);

                        _logger?.LogInformation($"[PythonServer] [StartAsync] Response generated.");

                        // Send Response
                        await _messagePipe.SendMessage(new PythonResponseMessage
                        {
                            Tensors = [response]
                        });

                        CallbackMessage("Generation Complete.");
                        _logger?.LogInformation($"[PythonServer] [StartAsync] Response sent.");
                    }
                    catch (EndOfStreamException)
                    { 
                        break;
                    }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, "[PythonServer] [StartAsync] An exception occurred");
                        throw;
                    }
                }
                _logger?.LogInformation($"[PythonServer] [StartAsync] Generate loop stopped.");
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
                    try
                    {
                        await _objectPipe.SendObject(progress, _cancellationTokenSource.Token);
                    }
                    catch (OperationCanceledException) { }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, $"[PythonServer] [ProcessProgressQueueAsync] - An exception occurred processing progress");
                    }
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
