using Microsoft.Extensions.DependencyInjection;
using System;
using System.Linq;
using System.Reflection;
using TensorStack.WPF.Controls;
using TensorStack.WPF.Services;

namespace TensorStack.WPF
{
    public static class Common
    {
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
    }
}

