using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Common;

namespace TensorStack.WPF.Services
{
    /// <summary>
    /// Service for handling the download of models and other dependancies
    /// </summary>
    /// <seealso cref="Amuserv.Common.Services.IDownloadService" />
    public sealed class DownloadService
    {
        private const int BufferSize = 131072;
        private readonly IHttpService _httpService;

        public DownloadService(IHttpService httpService)
        {
            _httpService = httpService;
        }

        /// <summary>
        /// Download as an asynchronous operation.
        /// </summary>
        /// <param name="fileUrl">The file URL.</param>
        /// <param name="outputFilename">The output filename.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task DownloadAsync(string fileUrl, string outputFilename, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var fileDownload = new FileDownloadResult
            {
                Url = fileUrl,
                FileName = outputFilename,
                Exists = File.Exists(outputFilename)
            };

            if (fileDownload.Exists)
                return;

            await DownloadAsync([fileDownload], _httpService.Client, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Downloads the files from a list of repository links, folder mapping by common prefix.
        /// </summary>
        /// <param name="repositoryFiles">The repository file urls.</param>
        /// <param name="outputDirectory">The output directory.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Queried file headers returned 0 bytes</exception>
        /// <exception cref="System.Exception">Model download failed: {ex.Message}</exception>
        public async Task DownloadAsync(List<string> repositoryFiles, string outputDirectory, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var downloadFiles = GetFileMapping(repositoryFiles, outputDirectory);
            if (downloadFiles.All(x => x.Exists))
                return;

            var remainingFiles = downloadFiles.Where(x => !x.Exists);
            await DownloadAsync(remainingFiles, _httpService.Client, progressCallback, cancellationToken);
        }


        /// <summary>
        /// Downloads the files from a list of repository links, folder mapping by common prefix.
        /// </summary>
        /// <param name="repositoryFiles">The repository files.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        public async Task DownloadAsync(Dictionary<string, string[]> repositoryFiles, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var downloadFiles = repositoryFiles.SelectMany(x => GetFileMapping(x.Value, x.Key));
            if (downloadFiles.All(x => x.Exists))
                return;

            var remainingFiles = downloadFiles.Where(x => !x.Exists);
            await DownloadAsync(remainingFiles, _httpService.Client, progressCallback, cancellationToken);
        }


        private static async Task DownloadAsync(IEnumerable<FileDownloadResult> downloadFiles, HttpClient httpClient, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            if (downloadFiles.All(x => x.Exists))
                return;

            var totalDownloadSize = await GetTotalSizeFromHeadersAsync(downloadFiles, httpClient, cancellationToken);
            if (totalDownloadSize == 0)
                throw new Exception("Queried file headers returned 0 bytes");

            var totalBytesRead = 0L;
            double bytesPerSecond = 0;
            var speedWindow = new Queue<(double Seconds, long TotalBytes)>();
            var totalStopwatch = Stopwatch.StartNew();
            foreach (var file in downloadFiles.Where(x => !x.Exists))
            {
                try
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    long existingFileSize = 0;
                    var tempFilename = $"{file.FileName}.download";
                    if (File.Exists(tempFilename))
                    {
                        var fileInfo = new FileInfo(tempFilename);
                        existingFileSize = fileInfo.Length;
                    }

                    Directory.CreateDirectory(Path.GetDirectoryName(tempFilename));
                    using (var fileStream = new FileStream(tempFilename, FileMode.OpenOrCreate, FileAccess.Write, FileShare.None))
                    {
                        if (existingFileSize > BufferSize)
                        {
                            totalBytesRead += existingFileSize;
                            fileStream.Seek(existingFileSize, SeekOrigin.Begin);
                            httpClient.DefaultRequestHeaders.Range = new System.Net.Http.Headers.RangeHeaderValue(existingFileSize, null);
                        }

                        using (var response = await httpClient.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                        {
                            response.EnsureSuccessStatusCode();

                            var fileBytesRead = existingFileSize;
                            var fileBuffer = new byte[BufferSize];
                            var fileSize = existingFileSize + response.Content.Headers.ContentLength ?? -1;

                            using (var contentStream = await response.Content.ReadAsStreamAsync(cancellationToken))
                            {
                                while (true)
                                {
                                    var timestamp = Stopwatch.GetTimestamp();
                                    cancellationToken.ThrowIfCancellationRequested();
                                    var readSize = await contentStream.ReadAsync(fileBuffer, cancellationToken);
                                    if (readSize == 0)
                                        break;

                                    await fileStream.WriteAsync(fileBuffer.AsMemory(0, readSize), cancellationToken);

                                    fileBytesRead += readSize;
                                    totalBytesRead += readSize;
                                    double elapsedSeconds = totalStopwatch.Elapsed.TotalSeconds;
                                    speedWindow.Enqueue((elapsedSeconds, totalBytesRead));
                                    while (speedWindow.Count > 1 && (elapsedSeconds - speedWindow.Peek().Seconds) > 1.0)
                                        speedWindow.Dequeue();

                                    bytesPerSecond = 0;
                                    if (speedWindow.Count > 1)
                                    {
                                        var oldest = speedWindow.Peek();
                                        double timeDelta = elapsedSeconds - oldest.Seconds;
                                        if (timeDelta > 0)
                                        {
                                            bytesPerSecond = (totalBytesRead - oldest.TotalBytes) / timeDelta;
                                        }
                                    }

                                    progressCallback?.Report(new DownloadProgress
                                    {
                                        FileSize = fileSize,
                                        FileBytes = fileBytesRead,
                                        FileProgress = fileBytesRead * 100.0 / fileSize,
                                        TotalSize = totalDownloadSize,
                                        TotalBytes = totalBytesRead,
                                        TotalProgress = totalBytesRead * 100.0 / totalDownloadSize,
                                        BytesSec = bytesPerSecond,
                                    });
                                }
                            }
                        }
                    }

                    // File Complete, Rename
                    File.Move(tempFilename, file.FileName, true);
                    file.Exists = File.Exists(file.FileName);
                }
                finally
                {
                    httpClient.DefaultRequestHeaders.Range = null;
                }
            }

            // Model Download Complete
            progressCallback?.Report(new DownloadProgress
            {
                FileSize = 0,
                FileBytes = 0,
                FileProgress = 100,
                TotalProgress = 100,
                TotalSize = totalDownloadSize,
                TotalBytes = totalDownloadSize,
                BytesSec = bytesPerSecond,
            });
        }


        /// <summary>
        /// Gets the total size from headers.
        /// </summary>
        /// <param name="fileList">The file list.</param>
        /// <param name="httpClient">The HTTP client.</param>
        /// <returns></returns>
        /// <exception cref="Exception">Failed to query file headers, {ex.Message}</exception>
        private static async Task<long> GetTotalSizeFromHeadersAsync(IEnumerable<FileDownloadResult> fileList, HttpClient httpClient, CancellationToken cancellationToken)
        {
            using (var semaphore = new SemaphoreSlim(1))
            {
                var tasks = new List<Task<long>>();
                foreach (var file in fileList)
                {
                    tasks.Add(Task.Run(async () =>
                    {
                        await semaphore.WaitAsync(cancellationToken);
                        try
                        {
                            using (var request = new HttpRequestMessage(HttpMethod.Head, file.Url))
                            using (var response = await httpClient.SendAsync(request, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                            {
                                if (response.StatusCode == System.Net.HttpStatusCode.MethodNotAllowed || response.StatusCode == System.Net.HttpStatusCode.NotImplemented)
                                {
                                    using (var fallbackRequest = new HttpRequestMessage(HttpMethod.Get, file.Url))
                                    using (var fallbackResponse = await httpClient.SendAsync(fallbackRequest, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                                    {
                                        fallbackResponse.EnsureSuccessStatusCode();
                                        return fallbackResponse.Content.Headers.ContentLength ?? 0L;
                                    }
                                }
                                response.EnsureSuccessStatusCode();
                                return response.Content.Headers.ContentLength ?? 0L;
                            }
                        }
                        catch { return 0L; }
                        finally { semaphore.Release(); }
                    }, cancellationToken));
                }

                var fileSizes = await Task.WhenAll(tasks);
                if (fileSizes.IsNullOrEmpty())
                    return 0L;

                return fileSizes.Sum();
            }
        }


        private static IEnumerable<FileDownloadResult> GetFileMapping(IEnumerable<string> urls, string outputDirectory)
        {
            var files = FileHelper.GetUrlFileMapping(urls, outputDirectory);
            return GetFileMapping(files);
        }


        private static IEnumerable<FileDownloadResult> GetFileMapping(Dictionary<string, string> files)
        {
            return files.Select(x => new FileDownloadResult
            {
                Url = x.Key,
                FileName = x.Value,
                Exists = File.Exists(x.Value)
            });
        }

    }



    public interface IHttpService
    {
        HttpClient Client { get; }
    }


    public record FileDownloadResult
    {
        public string Url { get; set; }
        public string FileName { get; set; }
        public bool Exists { get; set; }
    }


    public record DownloadProgress
    {
        public double FileProgress { get; set; }
        public double TotalProgress { get; set; }
        public long FileSize { get; set; }
        public long FileBytes { get; set; }
        public long TotalSize { get; set; }
        public long TotalBytes { get; set; }
        public double BytesSec { get; set; }
    }
}