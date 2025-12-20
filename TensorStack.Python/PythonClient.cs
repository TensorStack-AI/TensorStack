using Microsoft.Extensions.Logging;
using System;
using System.IO.Pipes;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Python.Config;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    public sealed class PythonClient : IDisposable
    {
        private readonly ILogger _logger;
        private readonly NamedPipeClientStream _objectPipe;
        private readonly NamedPipeClientStream _messagePipe;
        private CancellationTokenSource _cancellationTokenSource;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonClient"/> class.
        /// </summary>
        /// <param name="logger">The logger.</param>
        public PythonClient(ILogger logger = default)
        {
            _logger = logger;
            _objectPipe = new NamedPipeClientStream(".", ServerConfig.ObjectPipeName, PipeDirection.In, PipeOptions.Asynchronous);
            _messagePipe = new NamedPipeClientStream(".", ServerConfig.MessagePipeName, PipeDirection.InOut, PipeOptions.Asynchronous);
        }


        /// <summary>
        /// Start the client loop
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task StartAsync(IProgress<PythonProgress> progressCallback, CancellationToken cancellationToken)
        {
            // Connect Pipes
            await Task.WhenAll
            (
                _objectPipe.ConnectAsync(cancellationToken),
                _messagePipe.ConnectAsync(cancellationToken)
            );

            // Progress Loop
            _ = ProcessProgressQueueAsync(progressCallback);
        }


        /// <summary>
        /// Stop client and server.
        /// </summary>
        public async Task StopAsync()
        {
            await _cancellationTokenSource.SafeCancelAsync();
            try
            {
                await SendAsync(new PythonRequestMessage(PythonMessageType.Stop), CancellationToken.None);
            }
            catch (Exception ex)
            {
                _logger?.LogError(ex, "[] [] An exception occurred sending stop message.");
            }
        }


        /// <summary>
        /// Send as request to the Server
        /// </summary>
        /// <param name="request">The request.</param>
        public async Task<PythonResponseMessage> SendAsync(PythonRequestMessage request, CancellationToken cancellationToken)
        {
            // Send Request
            await _messagePipe.SendMessage(request, cancellationToken);

            // Receive Response
            return await _messagePipe.ReceiveMessage<PythonResponseMessage>(cancellationToken);
        }


        /// <summary>
        /// Process the progress queue
        /// </summary>
        /// <param name="statusPipe">The status pipe.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private async Task ProcessProgressQueueAsync(IProgress<PythonProgress> progressCallback)
        {
            using (_cancellationTokenSource = new CancellationTokenSource())
            {
                while (!_cancellationTokenSource.IsCancellationRequested)
                {
                    try
                    {
                        progressCallback?.Report(await _objectPipe.ReceiveObject<PythonProgress>(_cancellationTokenSource.Token));
                    }
                    catch (OperationCanceledException){ }
                    catch (Exception ex)
                    {
                        _logger?.LogError(ex, $"[PythonClient] [ProcessProgressQueueAsync] - An exception occurred processing progress");
                    }
                }
            }
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
