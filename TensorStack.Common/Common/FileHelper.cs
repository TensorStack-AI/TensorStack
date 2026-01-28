using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace TensorStack.Common.Common
{
    public static class FileHelper
    {
        public static bool DeleteFile(string filename)
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


        public static void DeleteFiles(params string[] filenames)
        {
            foreach (string filename in filenames)
            {
                DeleteFile(filename);
            }
        }


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


        public static string RandomFileName(string extension)
        {
            var ext = Path.HasExtension(extension) ? Path.GetExtension(extension) : extension;
            return $"{Path.GetFileNameWithoutExtension(Path.GetRandomFileName())}.{ext.Trim('.')}";
        }


        public static string RandomFileName(string directory, string extension)
        {
            Directory.CreateDirectory(directory);
            return Path.Combine(directory, RandomFileName(extension));
        }


        /// <summary>
        /// Gets the URL file mapping, mapping repository file structure to local directory
        /// </summary>
        /// <param name="sourceUrls">The source urls.</param>
        /// <param name="localDirectory">The local directory.</param>
        public static Dictionary<string, string> GetUrlFileMapping(IEnumerable<string> sourceUrls, string localDirectory)
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
                    .Skip(baseUrlSegmentLength)
                    .ToArray()) ?? string.Empty;
                var destination = Path.Combine(localDirectory, subFolder);
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
    }
}
