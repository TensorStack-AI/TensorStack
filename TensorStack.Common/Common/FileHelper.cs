using System;
using System.IO;

namespace TensorStack.Common.Common
{
    public static class FileHelper
    {
        public static bool DeleteFile(string filename)
        {
            try
            {
                if (File.Exists(filename))
                    return false;

                File.Delete(filename);
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


        public static string RandomFileName(string extension)
        {
            var ext = Path.HasExtension(extension) ? Path.GetExtension(extension) : extension;
            return $"{Path.GetFileNameWithoutExtension(Path.GetRandomFileName())}.{ext.Trim('.')}";
        }


        public static string RandomFileName(string directory, string extension)
        {
            return Path.Combine(directory, RandomFileName(extension));
        }
    }
}
