// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using TensorStack.WPF.Controls;
using static TensorStack.WPF.Native;

namespace TensorStack.WPF
{
    public static class WindowExtensions
    {
        public static void RegisterDisplayMonitor(this WindowBase window)
        {
            HwndSource.FromHwnd(window.Handle).AddHook(new HwndSourceHook(HookProc));
        }

        public static void RegisterDisplayMonitor(this DialogControl dialog)
        {
            HwndSource.FromHwnd(dialog.Handle).AddHook(new HwndSourceHook(HookProc));
        }

        private static IntPtr HookProc(IntPtr hwnd, int msg, IntPtr wParam, IntPtr lParam, ref bool handled)
        {
            if (msg == WM_GETMINMAXINFO)
            {
                // We need to tell the system what our size should be when maximized. Otherwise it will
                // cover the whole screen, including the task bar.
                MINMAXINFO mmi = (MINMAXINFO)Marshal.PtrToStructure(lParam, typeof(MINMAXINFO));

                // Adjust the maximized size and position to fit the work area of the correct monitor
                IntPtr monitor = MonitorFromWindow(hwnd, MONITOR_DEFAULTTONEAREST);

                if (monitor != IntPtr.Zero)
                {
                    MONITORINFO monitorInfo = new MONITORINFO();
                    monitorInfo.cbSize = Marshal.SizeOf(typeof(MONITORINFO));
                    GetMonitorInfo(monitor, ref monitorInfo);
                    RECT rcWorkArea = monitorInfo.rcWork;
                    RECT rcMonitorArea = monitorInfo.rcMonitor;

                    var x = rcWorkArea.Left - rcMonitorArea.Left;
                    var y = rcWorkArea.Top - rcMonitorArea.Top;
                    var width = rcWorkArea.Right - rcWorkArea.Left;
                    var height = rcWorkArea.Bottom - rcWorkArea.Top;

                    mmi.ptMaxPosition.X = x;
                    mmi.ptMaxPosition.Y = y;
                    mmi.ptMaxSize.X = width;
                    mmi.ptMaxSize.Y = height;
                    mmi.ptMaxTrackSize.X = width;
                    mmi.ptMaxTrackSize.Y = height;
                }

                Marshal.StructureToPtr(mmi, lParam, true);
            }

            return IntPtr.Zero;
        }


        public static Point GetMousePosition(Visual visualElement, bool dpiScale = false)
        {
            if (!GetCursorPos(out POINT point))
                return new Point(0, 0);

            if (dpiScale)
            {
                var source = PresentationSource.FromVisual(visualElement);
                if (source == null) 
                    return new Point(0, 0);

                double scaleX = point.X * source.CompositionTarget.TransformFromDevice.M11;
                double scaleY = point.Y * source.CompositionTarget.TransformFromDevice.M22;
                return visualElement.PointFromScreen(new Point(scaleX, scaleY));
            }

            return visualElement.PointFromScreen(new Point(point.X, point.Y));
        }
    }
}
