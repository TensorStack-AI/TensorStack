using System;
using System.Collections;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Net.Http;
using System.Threading;
using System.Threading.Tasks;
using Windows.UI.StartScreen;

namespace TensorStack.WPF.Services
{
    /// <summary>
    /// Service for handling the download of models and other dependancies
    /// </summary>
    /// <seealso cref="Amuserv.Common.Services.IDownloadService" />
    public sealed class DownloadService
    {
        private static readonly HttpClient HttpClient = new HttpClient();

        public async Task DownloadAsync(string fileUrl, string outputFilename, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
        {
            var fileDownload = new FileDownloadResult
            {
                Url = fileUrl,
                FileName = outputFilename,
            };

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

            await DownloadAsync(downloadFiles, progressCallback, cancellationToken);
        }


        private static async Task DownloadAsync(List<FileDownloadResult> downloadFiles, IProgress<DownloadProgress> progressCallback = default, CancellationToken cancellationToken = default)
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
        private static async Task<long> GetTotalSizeFromHeadersAsync(List<FileDownloadResult> fileList, HttpClient httpClient, CancellationToken cancellationToken)
        {
            var totalDownloadSize = 0L;
            var responseTasks = new Task<HttpResponseMessage>[fileList.Count];
            foreach (var file in fileList.Index())
            {
                responseTasks[file.Index] = httpClient.GetAsync(file.Item.Url, HttpCompletionOption.ResponseHeadersRead, cancellationToken);
            }

            var responses = await Task.WhenAll(responseTasks);
            foreach (var response in responses)
            {
                response.EnsureSuccessStatusCode();
                totalDownloadSize += response.Content.Headers.ContentLength ?? 0;
            }

            return totalDownloadSize;
        }


        /// <summary>
        /// Gets the length of the base URL segment.
        /// </summary>
        /// <param name="repositoryUrls">The repository urls.</param>
        /// <returns></returns>
        private static int GetBaseUrlSegmentLength(List<Uri> repositoryUrls)
        {
            var minUrlSegmentLength = repositoryUrls.Select(x => x.Segments.Length).Min();
            for (int i = 0; i < minUrlSegmentLength; i++)
            {
                if (repositoryUrls.Select(x => x.Segments[i]).Distinct().Count() > 1)
                {
                    return i;
                }
            }
            return minUrlSegmentLength;
        }


        private static List<FileDownloadResult> GetFileMapping(List<string> urls, string outputDirectory)
        {
            var files = new List<FileDownloadResult>();
            var repositoryUrls = urls.Select(x => new Uri(x)).ToList();
            var baseUrlSegmentLength = GetBaseUrlSegmentLength(repositoryUrls);
            foreach (var repositoryUrl in repositoryUrls)
            {
                var filename = repositoryUrl.Segments.Last().Trim('\\', '/');
                var subFolder = Path.Combine(repositoryUrl.Segments
                    .Where(x => x != repositoryUrl.Segments.Last())
                    .Select(x => x.Trim('\\', '/'))
                    .Skip(baseUrlSegmentLength)
                    .ToArray()) ?? string.Empty;
                var destination = Path.Combine(outputDirectory, subFolder);
                var destinationFile = Path.Combine(destination, filename);

                files.Add(new FileDownloadResult
                {
                    Url = repositoryUrl.OriginalString,
                    FileName = destinationFile,
                    Exists = File.Exists(destinationFile)
                });
            }
            return files;
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