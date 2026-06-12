using System;
using System.IO;
using System.Linq;
using System.Text.RegularExpressions;
using TensorStack.Common.Common;

namespace Amuse.App
{
    public static partial class HuggingFace
    {
        public static string GetCacheId(string repositoryUrl)
        {
            return $"models--{repositoryUrl.Replace("/", "--")}";
        }


        public static bool IsCheckpointInstalled(string modelDirectory, string checkpoint)
        {
            if (string.IsNullOrEmpty(checkpoint))
                return false;

            if (File.Exists(checkpoint))
                return true;

            if (!TryParseRepo(checkpoint, out var repositoryId))
                return false;

            var directory = Path.Combine(modelDirectory, GetCacheId(repositoryId));
            if (!Directory.Exists(directory))
                return false;

            var filename = Path.GetFileName(checkpoint);
            return Directory.EnumerateFiles(directory, filename, SearchOption.AllDirectories).Any();
        }


        public static bool IsLoraAdapterInstalled(string modelDirectory, string path, string weights)
        {
            if (string.IsNullOrEmpty(path) || string.IsNullOrEmpty(weights))
                return false;

            var adapter = Path.Combine(path, weights);
            if (File.Exists(adapter))
                return true;

            if (!TryParseRepo(path, out var repositoryId))
                return false;

            var directory = Path.Combine(modelDirectory, GetCacheId(repositoryId));
            if (!Directory.Exists(directory))
                return false;

            return FindCacheFile(directory, weights, SearchOption.AllDirectories) != null;
        }


        public static bool IsValidLink(string url)
        {
            if (string.IsNullOrWhiteSpace(url))
                return false;

            if (url.Split('/', StringSplitOptions.RemoveEmptyEntries).Length == 2)
                return true; // username/repository

            return HuggingFaceLinkRegex.IsMatch(url);
        }


        public static bool TryParseRepo(string input, out string repoId)
        {
            repoId = null;

            if (string.IsNullOrWhiteSpace(input))
                return false;

            if (input.Contains("/blob/main/"))
                input = input.Split("/blob/main/").FirstOrDefault();
            else if (input.Contains("/resolve/main/"))
                input = input.Split("/resolve/main/").FirstOrDefault();

            var match = HuggingFaceRepoRegex.Match(input.Trim());
            if (!match.Success)
                return false;

            repoId = match.Groups["repo"].Value;
            return true;
        }


        private static FileInfo FindCacheFile(string directory, string filename, SearchOption searchOption = SearchOption.AllDirectories)
        {
            var file = Directory.EnumerateFiles(directory, filename, searchOption).FirstOrDefault();
            if (string.IsNullOrEmpty(file))
                return default;

            return new FileInfo(file);
        }


        private static void DeleteCacheDirectory(string modelDirectory, string cacheId)
        {
            var cachePath = Path.Combine(modelDirectory, cacheId);
            var cacheLockPath = Path.Combine(modelDirectory, ".locks", cacheId);
            FileHelper.DeleteDirectory(cachePath);
            FileHelper.DeleteDirectory(cacheLockPath);
        }


        private static bool DeleteCacheFile(string filename)
        {
            try
            {
                if (!File.Exists(filename))
                    return true;

                var fileInfo = new FileInfo(filename);
                if (fileInfo.Exists)
                {
                    if (fileInfo.Attributes.HasFlag(FileAttributes.ReparsePoint))
                    {
                        var targetPath = fileInfo.LinkTarget;
                        if (!string.IsNullOrEmpty(targetPath))
                        {
                            var tagetFile = Path.GetFullPath(targetPath, fileInfo.DirectoryName);
                            if (File.Exists(tagetFile))
                            {
                                File.Delete(tagetFile);
                            }
                        }
                    }
                    File.Delete(filename);
                }

                return true;
            }
            catch (IOException) { return false; }
            catch (UnauthorizedAccessException) { return false; }
        }


        private static readonly Regex HuggingFaceLinkRegex = CreateHuggingFaceLinkRegex();
        private static readonly Regex HuggingFaceRepoRegex = CreateHuggingFaceRepoRegex();

        [GeneratedRegex(@"^https?:\/\/(www\.)?huggingface\.co\/(datasets\/|spaces\/)?[\w.-]+\/[\w.-]+", RegexOptions.IgnoreCase | RegexOptions.Compiled, "en-NZ")]
        private static partial Regex CreateHuggingFaceLinkRegex();

        [GeneratedRegex(@"^(?:https?:\/\/)?(?:www\.)?huggingface\.co\/(?<repo>[^\/\s]+\/[^\/\s]+)$|^(?<repo>[^\/\s]+\/[^\/\s]+)$", RegexOptions.IgnoreCase | RegexOptions.Compiled, "en-NZ")]
        private static partial Regex CreateHuggingFaceRepoRegex();
    }
}
