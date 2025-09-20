using System;
using System.Diagnostics;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Common;
using TensorStack.Common.Video;

namespace TensorStack.Video
{
    public class VideoService : IVideoService
    {
        private readonly IVideoConfiguration _configuration;

        /// <summary>
        /// Initializes a new instance of the <see cref="VideoService"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public VideoService(IVideoConfiguration configuration)
        {
            _configuration = configuration;
        }

        /// <summary>
        /// Get video information
        /// </summary>
        /// <param name="filename">The filename.</param>
        public async Task<VideoInfo> GetInfoAsync(string filename)
        {
            return await VideoManager.LoadVideoInfoAsync(filename);
        }


        /// <summary>
        /// Get the Video stream
        /// </summary>
        /// <param name="filename">The filename.</param>
        public async Task<VideoInputStream> GetStreamAsync(string filename)
        {
            return await VideoInputStream.CreateAsync(filename);
        }


        /// <summary>
        /// Saves the stream aync.
        /// </summary>
        /// <param name="videoInput">The video input.</param>
        /// <param name="videoFile">The video file.</param>
        /// <param name="frameProcessor">The frame processor.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<VideoInputStream> SaveStreamAync(VideoInputStream videoInput, string videoOutputFile, Func<VideoFrame, Task<VideoFrame>> frameProcessor, CancellationToken cancellationToken = default)
        {
            var videoFrames = videoInput.GetAsync(cancellationToken: cancellationToken);
            await VideoManager.WriteVideoStreamAsync(videoOutputFile, videoFrames, frameProcessor, _configuration.ReadBuffer, _configuration.ReadBuffer, _configuration.VideoCodec, cancellationToken: cancellationToken);
            await AddAudioAsync(videoOutputFile, videoInput.Filename, cancellationToken);
            return await VideoInputStream.CreateAsync(videoOutputFile);
        }



        private async Task AddAudioAsync(string target, string source, CancellationToken cancellationToken = default)
        {
            var tempFile = FileHelper.RandomFileName(_configuration.DirectoryTemp, target);
            var arguments = $"-i \"{target}\" -i \"{source}\" -c:v copy -c:a copy -map 0:v:0 -map 1:a:0 -y \"{tempFile}\"";

            try
            {
                using (var videoWriter = CreateProcess(arguments))
                {
                    videoWriter.Start();
                    await videoWriter.WaitForExitAsync(cancellationToken);
                }

                if (File.Exists(tempFile))
                    File.Move(tempFile, target, true);
            }
            finally
            {
                FileHelper.DeleteFile(tempFile);
            }
        }


        private Process CreateProcess(string arguments)
        {
            var process = new Process();
            process.StartInfo.FileName = _configuration.FFmpegPath;
            process.StartInfo.Arguments = arguments;
            process.StartInfo.RedirectStandardInput = true;
            process.StartInfo.UseShellExecute = false;
            process.StartInfo.CreateNoWindow = true;
            return process;
        }

    }
}
