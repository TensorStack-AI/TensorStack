using CSnakes.Runtime;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Threading.Tasks;
using TensorStack.Common.Common;
using TensorStack.Python.Common;
using TensorStack.Python.Config;

namespace TensorStack.Python
{
    /// <summary>
    /// PythonManager - Manage Python portable installation and virtual environment creation
    /// </summary>
    public class PythonManager
    {
        private readonly ILogger _logger;
        private readonly EnvironmentConfig _config;
        private readonly string _pythonPath;
        private readonly string _pipelinePath;
        private readonly string _pythonVersion = "3.12.10";

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonManager"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="logger">The logger.</param>
        public PythonManager(EnvironmentConfig config, ILogger logger = default)
        {
            _logger = logger;
            _config = config;
            _pythonPath = Path.GetFullPath(Path.Join(_config.Directory, "Python"));
            _pipelinePath = Path.GetFullPath(Path.Join(_config.Directory, "Pipelines"));
            CopyInternalPythonFiles();
            CopyInternalPipelineFiles();
        }


        public async Task<IPythonEnvironment> LoadAsync(IProgress<PipelineProgress> progressCallback = null)
        {
            return await LoadInternalAsync(progressCallback);
        }


        /// <summary>
        /// Creates the Python Virtual Environment.
        /// </summary>
        /// <param name="isRebuild">Delete and rebuild the environment</param>
        /// <param name="isReinstall">Delete and rebuild the environment and base Python installation</param>
        public Task<IPythonEnvironment> CreateAsync(bool isRebuild = false, bool isReinstall = false, IProgress<PipelineProgress> progressCallback = null)
        {
            return Task.Run(async () =>
            {
                await DownloadAsync(isReinstall, progressCallback);
                if (isReinstall || isRebuild)
                    await DeleteAsync();

                return await CreateInternalAsync(progressCallback);
            });
        }


        /// <summary>
        /// Delete an environment
        /// </summary>
        private async Task<bool> DeleteAsync()
        {
            var path = Path.Combine(_pipelinePath, $".{_config.Environment}");
            if (!Directory.Exists(path))
                return false;

            await Task.Run(() => Directory.Delete(path, true));
            return Exists();
        }


        /// <summary>
        /// Checks if a environment exists
        /// </summary>
        /// <param name="name">The name.</param>
        public bool Exists()
        {
            var path = Path.Combine(_pipelinePath, $".{_config.Environment}");
            return Directory.Exists(path);
        }


        /// <summary>
        /// Creates an environment.
        /// </summary>
        private async Task<IPythonEnvironment> CreateInternalAsync(IProgress<PipelineProgress> progressCallback = null)
        {
            var requirementsFile = Path.Combine(_pipelinePath, "requirements.txt");
            try
            {
                progressCallback.SendMessage($"Creating Python Virtual Environment (.{_config.Environment})");
                await File.WriteAllLinesAsync(requirementsFile, _config.Requirements);
                var environment = PythonEnvironmentHelper.CreateEnvironment(_config.Environment, _pythonPath, _pipelinePath, requirementsFile, _pythonVersion, _logger);
                progressCallback.SendMessage($"Python Virtual Environment Created");
                return environment;
            }
            finally
            {
                FileHelper.DeleteFile(requirementsFile);
            }
        }


        /// <summary>
        /// Load an existing Environment.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <returns>A Task&lt;IPythonEnvironment&gt; representing the asynchronous operation.</returns>
        /// <exception cref="System.Exception">Environment does not exist</exception>
        private async Task<IPythonEnvironment> LoadInternalAsync(IProgress<PipelineProgress> progressCallback = null)
        {
            if (!Exists())
                throw new Exception("Environment does not exist");

            return await Task.Run(() =>
            {
                progressCallback.SendMessage($"Loading Python Virtual Environment (.{_config.Environment})");
                var environment = PythonEnvironmentHelper.CreateEnvironment(_config.Environment, _pythonPath, _pipelinePath, _pythonVersion, _logger);
                progressCallback.SendMessage($"Python Virtual Environment Loaded");
                return environment;
            });
        }


        /// <summary>
        /// Downloads and installs Win-Python portable v3.12.10.
        /// </summary>
        /// <param name="reinstall">if set to <c>true</c> [reinstall].</param>
        private async Task DownloadAsync(bool reinstall, IProgress<PipelineProgress> progressCallback = null)
        {
            var subfolder = "WPy64-312100/python";
            var exePath = Path.Combine(_pythonPath, "python.exe");
            var downloadPath = Path.Combine(_pythonPath, "Winpython64-3.12.10.0dot.zip");
            var pythonUrl = "https://github.com/winpython/winpython/releases/download/15.3.20250425final/Winpython64-3.12.10.0dot.zip";
            if (reinstall)
            {
                progressCallback.SendMessage($"Reinstalling Python {_pythonVersion}...");
                if (File.Exists(downloadPath))
                    File.Delete(downloadPath);

                if (Directory.Exists(_pythonPath))
                    Directory.Delete(_pythonPath, true);

                progressCallback.SendMessage($"Python Uninstalled.");
            }

            // Create Python 
            Directory.CreateDirectory(_pythonPath);

            // Download Python
            if (!File.Exists(downloadPath))
            {
                progressCallback.SendMessage($"Download Python {_pythonVersion}...");
                using (var httpClient = new HttpClient())
                using (var response = await httpClient.GetAsync(pythonUrl))
                {
                    response.EnsureSuccessStatusCode();
                    using (var stream = new FileStream(downloadPath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        await response.Content.CopyToAsync(stream);
                    }
                }
                progressCallback.SendMessage("Python Download Complete.");
            }

            // Extract ZIP file
            if (!File.Exists(exePath))
            {
                progressCallback.SendMessage($"Installing Python {_pythonVersion}...");
                CopyInternalPythonFiles();
                using (var archive = ZipFile.OpenRead(downloadPath))
                {
                    foreach (var entry in archive.Entries)
                    {
                        if (entry.FullName.StartsWith(subfolder, StringComparison.OrdinalIgnoreCase))
                        {
                            var relativePath = entry.FullName.Replace('/', '\\').Substring(subfolder.Length);
                            if (string.IsNullOrWhiteSpace(relativePath))
                                continue;

                            var isDirectory = relativePath.EndsWith('\\');
                            var destinationPath = Path.Combine(_pythonPath, relativePath.TrimStart('\\').TrimEnd('\\'));
                            if (isDirectory)
                            {
                                Directory.CreateDirectory(destinationPath);
                                continue;
                            }
                            entry.ExtractToFile(destinationPath, overwrite: true);
                        }
                    }
                }
                progressCallback.SendMessage($"Python Install Complete.");
            }
        }

        /// <summary>
        /// Copies the internal python files.
        /// </summary>
        private void CopyInternalPythonFiles()
        {
            Directory.CreateDirectory(_pythonPath);
            CopyFiles(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Python"), _pythonPath);
        }


        /// <summary>
        /// Copies the internal pipeline files.
        /// </summary>
        private void CopyInternalPipelineFiles()
        {
            Directory.CreateDirectory(_pipelinePath);
            CopyFiles(Path.Combine(AppDomain.CurrentDomain.BaseDirectory, "Pipelines"), _pipelinePath);
        }


        /// <summary>
        /// Copies the files from source to target.
        /// </summary>
        /// <param name="sourcePath">The source path.</param>
        /// <param name="targetPath">The target path.</param>
        private static void CopyFiles(string sourcePath, string targetPath)
        {
            foreach (var dirPath in Directory.GetDirectories(sourcePath, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(dirPath.Replace(sourcePath, targetPath));

            foreach (var sourceFile in Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories))
            {
                var targetFile = sourceFile.Replace(sourcePath, targetPath);
                if (!File.Exists(targetFile) || File.GetLastWriteTimeUtc(sourceFile) > File.GetLastWriteTimeUtc(targetFile))
                {
                    File.Copy(sourceFile, targetFile, true);
                }
            }
        }
    }
}
