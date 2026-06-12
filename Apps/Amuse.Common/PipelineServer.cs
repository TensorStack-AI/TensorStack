using Amuse.Common.Config;
using Amuse.Common.Message;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using TensorStack.Common;

namespace Amuse.Common
{
    public abstract class PipelineServer : IDisposable
    {
        private readonly NamedPipeServerStream _commandChannel;
        private readonly NamedPipeServerStream _pipelineChannel;
        private readonly NamedPipeServerStream _progressChannel;
        private readonly Channel<PipelineProgress> _progressQueue;
        private RequestType _pipelineState;


        public PipelineServer(ServerConfig config, ILogger logger)
        {
            Logger = logger;
            Config = config;
            _progressQueue = Channel.CreateUnbounded<PipelineProgress>();
            _progressChannel = new NamedPipeServerStream(Config.ChannelProgress, PipeDirection.Out, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous, Config.ChunkSize, Config.ChunkSize);
            _commandChannel = new NamedPipeServerStream(Config.ChannelCommand, PipeDirection.InOut, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous, Config.ChunkSize, Config.ChunkSize);
            _pipelineChannel = new NamedPipeServerStream(Config.ChannelPipeName, PipeDirection.InOut, NamedPipeServerStream.MaxAllowedServerInstances, PipeTransmissionMode.Byte, PipeOptions.Asynchronous, Config.ChunkSize, Config.ChunkSize);
        }

        protected ILogger Logger { get; }
        protected ServerConfig Config { get; }
        protected CancellationTokenSource PipelineCancellation { get; set; }
        protected RequestType PipelineState => _pipelineState;


        /// <summary>
        /// Start the Server loop
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task StartAsync(CancellationToken cancellationToken = default)
        {
            await WaitForConnectionAsync(cancellationToken);

            _ = StartProgressChannelAsync(cancellationToken);
            _ = StartCommandChannelAsync(cancellationToken);
            await StartPipelineChannelAsync(cancellationToken);
            Logger.LogInformation($"[PipelineServer] [Start] Generate loop stopped.");
        }


        /// <summary>
        /// Wait for connection.
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task WaitForConnectionAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation($"[PipelineServer] [WaitForConnection] Waiting for connection...");
            await Task.WhenAll
            (
                _progressChannel.WaitForConnectionAsync(cancellationToken),
                _commandChannel.WaitForConnectionAsync(cancellationToken),
                _pipelineChannel.WaitForConnectionAsync(cancellationToken)
            );
            Logger.LogInformation($"[PipelineServer] [WaitForConnection] Client connected.");
        }


        /// <summary>
        /// Start pipeline channel
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task StartPipelineChannelAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation($"[PipelineServer] [PipelineChannel] Start pipeline channel.");

            _pipelineState = RequestType.Stop;
            await ChannelOpenedAsync();
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    Logger.LogInformation($"[PipelineServer] [PipelineChannel] Waiting for request.");
                    var request = await _pipelineChannel.ReceiveMessage<PipelineRequest>(cancellationToken);

                    Logger.LogInformation($"[PipelineServer] [PipelineChannel] {request.Type} request received.");
                    if (request.Type == RequestType.Stop)
                    {
                        await StopServerAsync(request, cancellationToken);
                        _pipelineState = RequestType.Stop;
                    }
                    else if (request.Type == RequestType.Start && _pipelineState == RequestType.Stop)
                    {
                        await StartServerAsync(request, cancellationToken);
                        _pipelineState = RequestType.Start;
                    }
                    else if (request.Type == RequestType.Create && _pipelineState == RequestType.Start)
                    {
                        await CreatePipelineAsync(request, cancellationToken);
                        _pipelineState = RequestType.Create;
                    }
                    else
                    {
                        if (_pipelineState == RequestType.Create)
                        {
                            if (request.Type == RequestType.Load)
                            {
                                await LoadPipelineAsync(request, cancellationToken);
                            }
                            else if (request.Type == RequestType.Reload)
                            {
                                await ReloadPipelineAsync(request, cancellationToken);
                            }
                            else if (request.Type == RequestType.Unload)
                            {
                                await UnloadPipelineAsync(request, cancellationToken);
                            }
                            else if (request.Type == RequestType.Run)
                            {
                                await RunPipelineAsync(request, cancellationToken);
                            }
                        }
                    }

                    if (_pipelineState == RequestType.Stop)
                        break;
                }
                catch (EndOfStreamException)
                {
                    break;
                }
                catch (Exception ex)
                {
                    Logger.LogError(ex, "[PipelineServer] [PipelineChannel] An unexpected exception occurred");
                    break;
                }
            }

            await ChannelClosedAsync();
            Logger.LogInformation($"[PipelineServer] [PipelineChannel] Pipeline channel closed.");
        }


        /// <summary>
        /// Start command channel
        /// </summary>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task StartCommandChannelAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation($"[PipelineServer] [CommandChannel] Start command channel.");
            while (!cancellationToken.IsCancellationRequested)
            {
                try
                {
                    Logger.LogInformation($"[PipelineServer] [CommandChannel] Waiting for command...");
                    var commandMessage = await _commandChannel.ReceiveObject<CommandRequest>(cancellationToken);
                    if (commandMessage == null)
                        continue;

                    Logger.LogInformation("[PipelineServer] [CommandChannel] Received {Type} command.", commandMessage.Type);
                    if (commandMessage.Type == CommandRequestType.Cancel)
                        await PipelineCancellation.SafeCancelAsync();

                    await _commandChannel.SendObject(new CommandResponse(), cancellationToken);
                    Logger.LogInformation("[PipelineServer] [CommandChannel] Processed {Type} command.", commandMessage.Type);
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    Logger.LogError(ex, $"[PipelineServer] [CommandChannel] - An exception occurred receiving command.");
                    await _commandChannel.SendObject(new CommandResponse(ex), cancellationToken);
                }
            }
            Logger.LogInformation($"[PipelineServer] [CommandChannel] Close command channel.");
        }


        /// <summary>
        /// Process the progress queue
        /// </summary>
        /// <param name="progressQueue">The progress queue.</param>
        protected async Task StartProgressChannelAsync(CancellationToken cancellationToken)
        {
            Logger.LogInformation($"[PipelineServer] [ProgressChannel] Start progress channel.");
            await foreach (var progressMessage in _progressQueue.Reader.ReadAllAsync(cancellationToken))
            {
                try
                {
                    await _progressChannel.SendObject(progressMessage, cancellationToken);
                }
                catch (OperationCanceledException) { }
                catch (Exception ex)
                {
                    Logger.LogError(ex, $"[PipelineServer] [ProgressChannel] - An exception occurred processing progress.");
                }
            }
            Logger.LogInformation($"[PipelineServer] [ProgressChannel] Close progress channel.");
        }


        /// <summary>
        /// Start the server
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task StartServerAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            await _pipelineChannel.SendResponse(cancellationToken);
            Logger.LogInformation($"[PipelineServer] [StartServer] Server started.");
        }


        /// <summary>
        /// Stop the server
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected async Task StopServerAsync(PipelineRequest request, CancellationToken cancellationToken)
        {
            await _pipelineChannel.SendResponse(cancellationToken);
            Logger.LogInformation($"[PipelineServer] [StopServer] Server stopped.");
        }


        protected async Task SendResponse(CancellationToken cancellationToken)
        {
            await _pipelineChannel.SendResponse(cancellationToken);
        }


        protected async Task SendMessage<T>(T message, CancellationToken cancellationToken) where T : IPipelineMessage
        {
            await _pipelineChannel.SendMessage(message, cancellationToken);
        }

        protected async Task SendException(Exception exception, CancellationToken cancellationToken)
        {
            await _pipelineChannel.SendMessage(new PipelineResponse(exception), cancellationToken);
        }


        protected async Task QueueProgress(PipelineProgress progress)
        {
            await _progressQueue.Writer.WriteAsync(progress);
        }

        /// <summary>
        /// Called when the Channel is opened.
        /// </summary>
        /// <returns>Task.</returns>
        protected abstract Task ChannelOpenedAsync();


        /// <summary>
        /// Called when the Channel is closed.
        /// </summary>
        /// <returns>Task.</returns>
        protected abstract Task ChannelClosedAsync();


        /// <summary>
        /// Create Pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected abstract Task CreatePipelineAsync(PipelineRequest request, CancellationToken cancellationToken);


        /// <summary>
        /// Loads the pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected abstract Task LoadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken);


        /// <summary>
        /// Reload the pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected abstract Task ReloadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken);


        /// <summary>
        /// Unloads the pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected abstract Task UnloadPipelineAsync(PipelineRequest request, CancellationToken cancellationToken);


        /// <summary>
        /// Runs the pipeline
        /// </summary>
        /// <param name="request">The request.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        protected abstract Task RunPipelineAsync(PipelineRequest request, CancellationToken cancellationToken);


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            PipelineCancellation?.SafeCancel();
            PipelineCancellation?.Dispose();
            _progressChannel?.Dispose();
            _commandChannel?.Dispose();
            _pipelineChannel?.Dispose();
            GC.SuppressFinalize(this);
        }
    }
}
