using CSnakes.Runtime;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.IO.Compression;
using System.Net.Http;
using System.Threading.Tasks;
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
        private readonly IProgress<PipelineProgress> _progressCallback;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonManager"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="logger">The logger.</param>
        public PythonManager(EnvironmentConfig config, IProgress<PipelineProgress> progressCallback = null, ILogger logger = default)
        {
            _logger = logger;
            _config = config;
            _progressCallback = progressCallback;
            _pythonPath = Path.GetFullPath(Path.Join(_config.Directory, "Python"));
            _pipelinePath = Path.GetFullPath(Path.Join(_config.Directory, "Pipelines"));
            CopyInternalPipelineFiles();
        }


        /// <summary>
        /// Creates the Python Virtual Environment.
        /// </summary>
        /// <param name="isRebuild">Delete and rebuild the environment</param>
        /// <param name="isReinstall">Delete and rebuild the environment and base Python installation</param>
        public Task<IPythonEnvironment> CreateEnvironmentAsync(bool isRebuild = false, bool isReinstall = false)
        {
            return Task.Run(async () =>
            {
                await DownloadAsync(isReinstall);
                if (isReinstall || isRebuild)
                    await DeleteAsync();

                return await CreateAsync();
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
            return Exists(path);
        }


        /// <summary>
        /// Checks if a environment exists
        /// </summary>
        /// <param name="name">The name.</param>
        private bool Exists(string name)
        {
            var path = Path.Combine(_pipelinePath, $".{name}");
            return Directory.Exists(path);
        }


        /// <summary>
        /// Creates the environment.
        /// </summary>
        private async Task<IPythonEnvironment> CreateAsync()
        {
            var exists = Exists(_config.Environment);
            CallbackMessage($"{(exists ? "Loading" : "Creating")} Python Virtual Environment (.{_config.Environment})");
            var requirementsFile = Path.Combine(_pipelinePath, "requirements.txt");
            await File.WriteAllLinesAsync(requirementsFile, _config.Requirements);
            var environment = PythonEnvironmentHelper.CreateEnvironment(_config.Environment, _pythonPath, _pipelinePath, requirementsFile, _pythonVersion, _logger);
            CallbackMessage($"Python Virtual Environment {(exists ? "Loaded" : "Created")}.");
            return environment;
        }


        /// <summary>
        /// Downloads and installs Win-Python portable v3.12.10.
        /// </summary>
        /// <param name="reinstall">if set to <c>true</c> [reinstall].</param>
        private async Task DownloadAsync(bool reinstall)
        {
            var subfolder = "WPy64-312100/python";
            var exePath = Path.Combine(_pythonPath, "python.exe");
            var downloadPath = Path.Combine(_pythonPath, "Winpython64-3.12.10.0dot.zip");
            var pythonUrl = "https://github.com/winpython/winpython/releases/download/15.3.20250425final/Winpython64-3.12.10.0dot.zip";
            if (reinstall)
            {
                CallbackMessage($"Reinstalling Python {_pythonVersion}...");
                if (File.Exists(downloadPath))
                    File.Delete(downloadPath);

                if (Directory.Exists(_pythonPath))
                    Directory.Delete(_pythonPath, true);

                CallbackMessage($"Python Uninstalled.");
            }

            // Create Python 
            Directory.CreateDirectory(_pythonPath);

            // Download Python
            if (!File.Exists(downloadPath))
            {
                CallbackMessage($"Download Python {_pythonVersion}...");
                using (var httpClient = new HttpClient())
                using (var response = await httpClient.GetAsync(pythonUrl))
                {
                    response.EnsureSuccessStatusCode();
                    using (var stream = new FileStream(downloadPath, FileMode.Create, FileAccess.Write, FileShare.None))
                    {
                        await response.Content.CopyToAsync(stream);
                    }
                }
                CallbackMessage("Python Download Complete.");
            }

            // Extract ZIP file
            if (!File.Exists(exePath))
            {
                CallbackMessage($"Installing Python {_pythonVersion}...");
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
                CallbackMessage($"Python Install Complete.");
            }
        }


        /// <summary>
        /// Send a callback message.
        /// </summary>
        /// <param name="message">The message.</param>
        private void CallbackMessage(string message)
        {
            _progressCallback?.Report(new PipelineProgress
            {
                Message = message,
                Process = "Initialize"
            });
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
            foreach (string dirPath in Directory.GetDirectories(sourcePath, "*", SearchOption.AllDirectories))
                Directory.CreateDirectory(dirPath.Replace(sourcePath, targetPath));

            foreach (string newPath in Directory.GetFiles(sourcePath, "*.*", SearchOption.AllDirectories))
                File.Copy(newPath, newPath.Replace(sourcePath, targetPath), true);
        }
    }
}
