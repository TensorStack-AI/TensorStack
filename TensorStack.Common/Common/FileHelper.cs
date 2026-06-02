using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TensorStack.Common.Common
{
    public static class FileHelper
    {
        /// <summary>
        /// Deletes the file.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns><c>true</c> if deleted, <c>false</c> otherwise.</returns>
        public static bool DeleteFile(string filename)
        {
            return TryDelete(filename);
        }


        /// <summary>
        /// Deletes the files.
        /// </summary>
        /// <param name="filenames">The filenames.</param>
        public static void DeleteFiles(params string[] filenames)
        {
            foreach (string filename in filenames)
            {
                TryDelete(filename);
            }
        }


        /// <summary>
        /// Queues the file for deletion files with retry. (5 retries, 500ms delay)
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <returns><c>true</c> if queued, <c>false</c> otherwise.</returns>
        public static bool QueueDeleteFile(string filename)
        {
            try
            {
                if (!File.Exists(filename))
                    return false;

                FileQueue.Delete(filename);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }


        /// <summary>
        /// Queues the files for deletion files with retry. (5 retries, 500ms delay)
        /// </summary>
        /// <param name="filenames">The filenames.</param>
        public static void QueueDeleteFiles(params string[] filenames)
        {
            foreach (string filename in filenames)
            {
                QueueDeleteFile(filename);
            }
        }


        /// <summary>
        /// Deletes the directory.
        /// </summary>
        /// <param name="directory">The directory.</param>
        /// <param name="recursive">if set to <c>true</c> recursive delete.</param>
        public static bool DeleteDirectory(string directory, bool recursive = true)
        {
            try
            {
                if (!Directory.Exists(directory))
                    return false;

                Directory.Delete(directory, recursive);
                return true;
            }
            catch (Exception)
            {
                return false;
            }
        }


        /// <summary>
        /// Genetare a random filename
        /// </summary>
        /// <param name="extension">The extension.</param>
        public static string RandomFileName(string extension)
        {
            var ext = Path.HasExtension(extension) ? Path.GetExtension(extension) : extension;
            return $"{Path.GetFileNameWithoutExtension(Path.GetRandomFileName())}.{ext.Trim('.')}";
        }


        /// <summary>
        /// Genetare a random filename
        /// </summary>
        /// <param name="directory">The directory.</param>
        /// <param name="extension">The extension.</param>
        public static string RandomFileName(string directory, string extension)
        {
            Directory.CreateDirectory(directory);
            return Path.Combine(directory, RandomFileName(extension));
        }


        /// <summary>
        /// Finds the file in the specified folder tree.
        /// </summary>
        /// <param name="directory">The directory.</param>
        /// <param name="filename">The filename.</param>
        /// <param name="searchOption">The search option.</param>
        public static FileInfo FindFile(string directory, string filename, SearchOption searchOption = SearchOption.AllDirectories)
        {
            var file = Directory.EnumerateFiles(directory, filename, searchOption).FirstOrDefault();
            if (string.IsNullOrEmpty(file))
                return default;

            return new FileInfo(file);
        }


        /// <summary>
        /// Determines whether the directory empty
        /// </summary>
        /// <param name="directory">The directory.</param>
        public static bool IsDirectoryEmpty(string directory)
        {
            if (!Directory.Exists(directory))
                return false;

            return !Directory.EnumerateFileSystemEntries(directory).Any();
        }


        public static Dictionary<string, string> GetFileMapping(IEnumerable<string> sourceUrls, string localDirectory)
        {
            var files = new Dictionary<string, string>();
            foreach (var sourceUrl in sourceUrls)
            {
                files.Add(sourceUrl, Path.Combine(localDirectory, Path.GetFileName(sourceUrl)));
            }
            return files;
        }


        /// <summary>
        /// Gets the URL file mapping, mapping repository file structure to local directory
        /// </summary>
        /// <param name="sourceUrls">The source urls.</param>
        /// <param name="localDirectory">The local directory.</param>
        /// <param name="replacement">The part path replacement.</param>
        public static Dictionary<string, string> GetUrlFileMapping(IEnumerable<string> sourceUrls, string localDirectory, string replacement = null, int skipSegments = 0)
        {
            var files = new Dictionary<string, string>();
            var repositoryUrls = sourceUrls.Select(x => new Uri(x));
            var baseUrlSegmentLength = GetBaseUrlSegmentLength(repositoryUrls);
            foreach (var repositoryUrl in repositoryUrls)
            {
                var filename = repositoryUrl.Segments.Last().Trim('\\', '/');
                var subFolder = Path.Combine(repositoryUrl.Segments
                    .Where(x => x != repositoryUrl.Segments.Last())
                    .Select(x => x.Trim('\\', '/'))
                    .Skip(baseUrlSegmentLength + skipSegments)
                    .ToArray()) ?? string.Empty;
                var destination = string.IsNullOrEmpty(replacement)
                   ? Path.Combine(localDirectory, subFolder)
                   : Path.Combine(localDirectory, subFolder.Replace(replacement, null));
                var destinationFile = Path.Combine(destination, filename);

                files.Add(repositoryUrl.OriginalString, destinationFile);
            }
            return files;
        }


        /// <summary>
        /// Gets the length of the base URL segment.
        /// </summary>
        /// <param name="repositoryUrls">The repository urls.</param>
        /// <returns></returns>
        private static int GetBaseUrlSegmentLength(IEnumerable<Uri> repositoryUrls)
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
    }
}
