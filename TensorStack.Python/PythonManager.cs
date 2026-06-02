using CSnakes.Runtime;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
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
        private readonly string _baseDirectory;
        private readonly PythonVersion _pythonVersion;

        /// <summary>
        /// Initializes a new instance of the <see cref="PythonManager"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        /// <param name="baseDirectory">The base directory.</param>
        /// <param name="logger">The logger.</param>
        public PythonManager(EnvironmentConfig config, string baseDirectory, ILogger logger = default)
        {
            _logger = logger;
            _config = config;
            _pythonVersion = GetPythonVresion(config.PythonVersion);
            _baseDirectory = Path.GetFullPath(baseDirectory);
            _pythonPath = Path.GetFullPath(Path.Join(_config.Directory, _pythonVersion.Folder));
            _pipelinePath = Path.GetFullPath(Path.Join(_config.Directory, "Pipelines"));
            CopyInternalPythonFiles();
            CopyInternalPipelineFiles();
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="PythonManager"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public PythonManager(EnvironmentConfig config, ILogger logger = default)
            : this(config, AppDomain.CurrentDomain.BaseDirectory, logger) { }


        /// <summary>
        /// Load an existing environment.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        public async Task<IPythonEnvironment> LoadAsync(IProgress<PipelineProgress> progressCallback = null)
        {
            return await LoadInternalAsync(progressCallback);
        }


        /// <summary>
        /// Creates the Python Virtual Environment.
        /// If the environment already exists it is loaded after package manager is run (update)
        /// </summary>
        /// <param name="isRebuild">Delete and rebuild the environment</param>
        /// <param name="isReinstall">Delete and rebuild the environment and base Python installation</param>
        public Task<IPythonEnvironment> CreateAsync(EnvironmentMode mode, IProgress<PipelineProgress> progressCallback = null)
        {
            return Task.Run(async () =>
            {
                var isRebuild = mode == EnvironmentMode.Rebuild;
                var isReinstall = mode == EnvironmentMode.Reinstall;
                await DownloadAsync(isReinstall, progressCallback);
                if (isReinstall || isRebuild)
                    await DeleteAsync();

                return await CreateInternalAsync(mode, progressCallback);
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
        private async Task<IPythonEnvironment> CreateInternalAsync(EnvironmentMode environmentMode, IProgress<PipelineProgress> progressCallback = null)
        {
            var requirementsFile = Path.Combine(_pipelinePath, "requirements.txt");
            try
            {
                progressCallback.SendMessage($"{environmentMode} Python Virtual Environment (.{_config.Environment})");
                await File.WriteAllLinesAsync(requirementsFile, _config.Requirements);
                var environment = PythonEnvironmentHelper.CreateEnvironment(_config.Environment, _pythonPath, _pipelinePath, requirementsFile, _pythonVersion.Version, _logger);
                progressCallback.SendMessage($"Python Virtual Environment Configured.");
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
                var environment = PythonEnvironmentHelper.CreateEnvironment(_config.Environment, _pythonPath, _pipelinePath, _pythonVersion.Version, _logger);
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
            var subfolder = $"{_pythonVersion.ZipFolder}/python";
            var exePath = Path.Combine(_pythonPath, "python.exe");
            var downloadPath = Path.Combine(_pythonPath, _pythonVersion.FileName);
            var pythonUrl = _pythonVersion.Link;
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
                            else
                            {
                                Directory.CreateDirectory(Path.GetDirectoryName(destinationPath));
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
            CopyFiles(Path.Combine(_baseDirectory, "Python"), _pythonPath);
        }


        /// <summary>
        /// Copies the internal pipeline files.
        /// </summary>
        private void CopyInternalPipelineFiles()
        {
            Directory.CreateDirectory(_pipelinePath);
            CopyFiles(Path.Combine(_baseDirectory, "Pipelines"), _pipelinePath);
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


        /// <summary>
        /// Gets the python vresion.
        /// </summary>
        /// <param name="versionId">The version identifier.</param>
        /// <exception cref="System.ArgumentException">Invalid PythonVersion '${versionId}', Expected: 3.12, 3.13 or 3.14</exception>
        private static PythonVersion GetPythonVresion(string versionId)
        {
            if (_supportedVersions.TryGetValue(versionId, out PythonVersion pythonVersion))
                return pythonVersion;

            throw new ArgumentException($"Invalid PythonVersion '${versionId}', Expected: 3.12, 3.13 or 3.14");
        }


        /// <summary>
        /// The supported Python versions
        /// </summary>
        private readonly static Dictionary<string, PythonVersion> _supportedVersions = new Dictionary<string, PythonVersion>
        {
            {"3.12", new PythonVersion("3.12.10", "https://github.com/winpython/winpython/releases/download/15.3.20250425final/Winpython64-3.12.10.0dot.zip", "Winpython64-3.12.10.0dot.zip", "WPy64-312100", "Python") },
            {"3.13", new PythonVersion("3.13.13", "https://github.com/winpython/winpython/releases/download/17.4.20260511final/WinPython64-3.13.13.0dot.zip", "WinPython64-3.13.13.0dot.zip", "WPy64-313130", "Python313") },
            {"3.14", new PythonVersion("3.14.5", "https://github.com/winpython/winpython/releases/download/17.4.20260511final/WinPython64-3.14.5.0dot.zip", "WinPython64-3.14.5.0dot.zip", "WPy64-31450", "Python314") }
        };
    }

    public record PythonVersion(string Version, string Link, string FileName, string ZipFolder, string Folder);
}
