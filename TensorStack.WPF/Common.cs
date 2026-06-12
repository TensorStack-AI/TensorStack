using Microsoft.Extensions.DependencyInjection;
using System;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Reflection;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.WPF
{
    public static class Common
    {
        public const double DragDistance = 20.0;

        public readonly static string[] TextFileExtensions = [".txt"];
        public const string TextFileFilter = "Text Files|*.txt;|All Files|*.*";

        public readonly static string[] ImageFileExtensions = [".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff"];
        public const string ImageFileFilter = "Image Files|*.bmp;*.jpg;*.jpeg;*.png;*.gif;*.tif;*.tiff|All Files|*.*";

        public readonly static string[] AudioFileExtensions = [".wav", ".mp3", ".aac", ".flac", ".m4a", ".ogg", ".mp4", ".mov", ".mkv", ".webm",];
        public const string AudioFileFilter = "Audio Files|*.mp3;*.wav;*.flac;*.m4a;*.aac;*.ogg;*.mp4;*.mov;*.mkv;*.webm|All Files|*.*";

        public readonly static string[] VideoFileExtensions = [".mp4", ".gif"];
        public const string VideoFileFilter = "Videos Files|*.mp4;*.gif;|All Files|*.*;";


        public static WindowMainBase GetMainWindow(this IServiceProvider services)
        {
            return services.GetRequiredService<WindowMainBase>();
        }

        public static void UseWPFCommon(this IServiceProvider services)
        {
            services.GetRequiredService<DialogService>();
        }


        public static void AddWPFCommon<T>(this IServiceCollection services, DefaultUIConfiguration configuration) where T : WindowMainBase
        {
            services.AddWPFCommon<T, DefaultUIConfiguration>(configuration);
        }


        public static void AddWPFCommon<T, C>(this IServiceCollection services, C configuration) where T : WindowMainBase where C : class, IUIConfiguration
        {
            var types = Assembly.GetExecutingAssembly().GetTypes().ToList();
            types.AddRange(Assembly.GetAssembly(typeof(T)).GetTypes());

            // Register Configuration
            services.AddSingleton<C>(configuration);
            var interfaces = typeof(C).GetInterfaces();
            foreach (var alias in interfaces)
            {
                services.AddSingleton(alias, sp => sp.GetRequiredService<C>());
            }

            // Register Services
            services.AddSingleton<DialogService>();
            services.AddSingleton<DownloadService>();
            services.AddSingleton<ComponentService>();
            services.AddSingleton<NavigationService>();

            // Register WindowBase
            services.AddSingleton<WindowMainBase, T>();

            // Register ViewControl (Singleton Only)
            foreach (var view in types.Where(type => typeof(ViewControl).IsAssignableFrom(type) && !type.IsAbstract))
            {
                services.AddSingleton(typeof(IViewControl), view);
            }

            // Register DialogControl
            foreach (var dialog in types.Where(type => type.BaseType == typeof(DialogControl)))
            {
                services.AddControl(dialog);
            }

            // Register Components
            foreach (var component in types.Where(type => type.BaseType == typeof(Component)))
            {
                services.AddControl(component);
            }
        }


        private static void AddControl(this IServiceCollection services, Type controlType)
        {
            if (controlType.IsSingletonControl())
            {
                services.AddSingleton(controlType);
                return;
            }
            services.AddTransient(controlType);
        }


        private static bool IsSingletonControl(this Type controlType)
        {
            return controlType.GetInterfaces().Contains(typeof(ILifetimeSingleton));
        }


        public static string GetDisplayName(this Enum enumObj)
        {
            var fieldInfo = enumObj.GetType().GetField(enumObj.ToString());
            var attribArray = fieldInfo.GetCustomAttributes(false);
            if (attribArray.Length > 0)
            {
                foreach (var att in attribArray)
                {
                    if (att is DisplayAttribute display)
                        return display.Name ?? enumObj.ToString();
                    else if (att is System.ComponentModel.DescriptionAttribute desc)
                        return desc.Description;
                }
            }
            return enumObj.ToString();
        }


        public static string GetShortName(this Enum enumObj)
        {
            var fieldInfo = enumObj.GetType().GetField(enumObj.ToString());
            var attribArray = fieldInfo.GetCustomAttributes(false);
            if (attribArray.Length > 0)
            {
                foreach (var att in attribArray)
                {
                    if (att is DisplayAttribute display)
                        return display.ShortName ?? enumObj.ToString();
                }
            }
            return enumObj.ToString();
        }


        public static string GetDisplayDescription(this Enum enumObj)
        {
            var fieldInfo = enumObj.GetType().GetField(enumObj.ToString());
            var attribArray = fieldInfo.GetCustomAttributes(false);
            if (attribArray.Length > 0)
            {
                foreach (var att in attribArray)
                {
                    if (att is DisplayAttribute display)
                        return display.Description;
                }
            }
            return string.Empty;
        }


        /// <summary>
        /// Converts color mask to Black & White mask. Transparent to Black to All other colors to White
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="invert">if set to <c>true</c> invert Black & White.</param>
        /// <returns>WriteableBitmap.</returns>
        public static WriteableBitmap ToBlackWhiteMask(this BitmapSource source, bool invert)
        {
            if (source.Format != PixelFormats.Pbgra32)
                source = new FormatConvertedBitmap(source, PixelFormats.Pbgra32, null, 0);

            var writeableBitmap = new WriteableBitmap(source);
            writeableBitmap.Lock();

            unsafe
            {
                const int white = unchecked((int)0xFFFFFFFF);
                const int black = unchecked((int)0xFF000000);

                int* pBuffer = (int*)writeableBitmap.BackBuffer;
                int totalPixels = writeableBitmap.PixelWidth * writeableBitmap.PixelHeight;

                int onValue = invert ? black : white;
                int offValue = invert ? white : black;

                for (int i = 0; i < totalPixels; i++)
                {
                    pBuffer[i] = (pBuffer[i] == 0) ? offValue : onValue;
                }
            }

            writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, writeableBitmap.PixelWidth, writeableBitmap.PixelHeight));
            writeableBitmap.Unlock();
            writeableBitmap.Freeze();
            return writeableBitmap;
        }


        /// <summary>
        /// Converts to image to a Mask, Black to Transparent, All other colors to MaskColor
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="maskColor">Color of the mask.</param>
        public static WriteableBitmap ToColorMask(this BitmapSource source, Color maskColor)
        {
            if (source.Format != PixelFormats.Pbgra32)
                source = new FormatConvertedBitmap(source, PixelFormats.Pbgra32, null, 0);

            var writeableBitmap = new WriteableBitmap(source);
            writeableBitmap.Lock();

            unsafe
            {
                int* pBuffer = (int*)writeableBitmap.BackBuffer;
                int totalPixels = writeableBitmap.PixelWidth * writeableBitmap.PixelHeight;

                // Handle premultiplied alpha if maskColor is semi-transparent
                byte a = maskColor.A;
                byte r = (byte)((maskColor.R * a + 127) / 255);
                byte g = (byte)((maskColor.G * a + 127) / 255);
                byte b = (byte)((maskColor.B * a + 127) / 255);
                int maskPixel = (a << 24) | (r << 16) | (g << 8) | b;

                for (int i = 0; i < totalPixels; i++)
                {
                    int currentPixel = pBuffer[i];
                    byte pixB = (byte)(currentPixel & 0xFF);
                    byte pixG = (byte)((currentPixel >> 8) & 0xFF);
                    byte pixR = (byte)((currentPixel >> 16) & 0xFF);
                    byte pixA = (byte)((currentPixel >> 24) & 0xFF);

                    // Mask only if it has Alpha AND isn't Black, threshold to catch JPEG noise
                    bool isTransparent = pixA == 0;
                    bool isBlack = pixR < 10 && pixG < 10 && pixB < 10;
                    if (isTransparent || isBlack)
                    {
                        pBuffer[i] = 0; // Turn black/transparent background into empty space
                    }
                    else
                    {
                        pBuffer[i] = maskPixel; // Turn everything else (White/Colors) into mask color
                    }
                }
            }

            writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, writeableBitmap.PixelWidth, writeableBitmap.PixelHeight));
            writeableBitmap.Unlock();
            writeableBitmap.Freeze();
            return writeableBitmap;
        }


        public static RenderTargetBitmap CreateBitmap(this UIElement element, int width, int height)
        {
            var renderBitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
            element.Measure(new Size(width, height));
            element.Arrange(new Rect(new Size(width, height)));
            element.UpdateLayout();
            renderBitmap.Render(element);
            renderBitmap.Freeze();
            return renderBitmap;
        }
    }
}