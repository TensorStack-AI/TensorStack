using System;

namespace TensorStack.Image
{
    public static class Extensions
    {
        /// <summary>
        /// Normalizes to float.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.Single.</returns>
        public static float NormalizeToFloat(this byte value)
        {
            return (value / 255.0f) * 2.0f - 1.0f;
        }


        /// <summary>
        /// Denormalizes to byte.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <returns>System.Byte.</returns>
        public static byte DenormalizeToByte(this float value)
        {
            return (byte)Math.Round(Math.Clamp((value / 2.0 + 0.5) * 255.0, 0.0, 255.0));
        }
    }
}
