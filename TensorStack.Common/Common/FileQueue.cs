using System;
using System.Collections.Concurrent;
using System.IO;
using System.Threading;
using System.Threading.Tasks;

namespace TensorStack.Common.Common
{
    public static class FileQueue
    {
        private static readonly ConcurrentQueue<(string path, int retries)> _queue = new();
        private static readonly SemaphoreSlim _signal = new(0);
        private static readonly CancellationTokenSource _cts = new();
        private const int MaxRetries = 5;
        private const int RetryDelayMs = 500;


        static FileQueue()
        {
            Task.Run(Worker);
        }


        public static void Delete(string path)
        {
            _queue.Enqueue((path, 0));
            _signal.Release();
        }


        private static async Task Worker()
        {
            while (!_cts.IsCancellationRequested)
            {
                await _signal.WaitAsync(_cts.Token);

                if (!_queue.TryDequeue(out var item))
                    continue;

                if (TryDelete(item.path))
                    continue;

                if (item.retries < MaxRetries)
                {
                    await Task.Delay(RetryDelayMs);
                    _queue.Enqueue((item.path, item.retries + 1));
                    _signal.Release();
                }
            }
        }


        private static bool TryDelete(string filename)
        {
            try
            {
                if (!File.Exists(filename))
                    return true;

                File.Delete(filename);
                return true;
            }
            catch (IOException) { return false; }
            catch (UnauthorizedAccessException) { return false; }
        }


        public static void Shutdown()
        {
            _cts.Cancel();
            _signal.Release();
        }
    }
}
