using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Common;

namespace TensorStack.WPF.Services
{
    /// <summary>
    /// Service for handling the download of models and other dependancies
    /// </summary>
    /// <seealso cref="Amuserv.Common.Services.IDownloadService" />
    public sealed class DownloadService
    {
        private static readonly HttpClient HttpClient = new HttpClient();


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

            await DownloadAsync([fileDownload], progressCallback, cancellationToken);
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
            await DownloadAsync(remainingFiles, progressCallback, cancellationToken);
        }


        private static async Task DownloadAsync(IEnumerable<FileDownloadResult> downloadFiles, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            if (downloadFiles.All(x => x.Exists))
                return;

            var totalDownloadSize = await GetTotalSizeFromHeadersAsync(downloadFiles, HttpClient, cancellationToken);
            if (totalDownloadSize == 0)
                throw new Exception("Queried file headers returned 0 bytes");

            var totalBytesRead = 0L;
            var bytePerSecond = new Queue<double>();
            foreach (var file in downloadFiles.Where(x => !x.Exists))
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
                    if (existingFileSize > 0)
                    {
                        fileStream.Seek(existingFileSize, SeekOrigin.Begin);
                        HttpClient.DefaultRequestHeaders.Range = new System.Net.Http.Headers.RangeHeaderValue(existingFileSize, null);
                    }

                    using (var response = await HttpClient.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken))
                    {
                        response.EnsureSuccessStatusCode();

                        var fileBytesRead = 0;
                        var fileBuffer = new byte[32768];
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
                                var fileProgress = fileBytesRead * 100.0 / fileSize;
                                var totalProgressValue = totalBytesRead * 100.0 / totalDownloadSize;

                                bytePerSecond.Enqueue(readSize / Stopwatch.GetElapsedTime(timestamp).TotalSeconds);
                                if (bytePerSecond.Count > 500)
                                    bytePerSecond.Dequeue();

                                progressCallback?.Report(new DownloadProgress
                                {
                                    FileSize = fileSize,
                                    FileBytes = fileBytesRead,
                                    FileProgress = fileProgress,
                                    TotalSize = totalDownloadSize,
                                    TotalBytes = totalBytesRead,
                                    TotalProgress = totalProgressValue,
                                    BytesSec = bytePerSecond.Average(),
                                });
                            }
                        }
                    }
                }

                // File Complete, Rename
                File.Move(tempFilename, file.FileName, true);
                file.Exists = File.Exists(file.FileName);
            }

            // Model Download Complete
            progressCallback?.Report(new DownloadProgress
            {
                TotalProgress = 100,
                TotalSize = totalDownloadSize,
                TotalBytes = totalDownloadSize,
                BytesSec = bytePerSecond.Average(),
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
            var totalDownloadSize = 0L;
            var responseTasks = new List<Task<HttpResponseMessage>>();
            foreach (var file in fileList)
            {
                responseTasks.Add(httpClient.GetAsync(file.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken));
            }

            var responses = await Task.WhenAll(responseTasks);
            foreach (var response in responses)
            {
                response.EnsureSuccessStatusCode();
                totalDownloadSize += response.Content.Headers.ContentLength ?? 0;
            }

            return totalDownloadSize;
        }


        private static IEnumerable<FileDownloadResult> GetFileMapping(List<string> urls, string outputDirectory)
        {
            var files = FileHelper.GetUrlFileMapping(urls, outputDirectory);
            return files.Select(x => new FileDownloadResult
            {
                Url = x.Key,
                FileName = x.Value,
                Exists = File.Exists(x.Value)
            });
        }

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
        public int FileBytes { get; set; }
        public long TotalSize { get; set; }
        public long TotalBytes { get; set; }
        public double BytesSec { get; set; }
    }
}