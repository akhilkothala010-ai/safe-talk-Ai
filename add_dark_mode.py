import os
import glob

css_addition = """
        /* Dark Mode Overrides */
        html.dark body {
            background-color: #111827;
            color: #f3f4f6;
        }
        html.dark .hero-bg {
            background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        }
        html.dark .glass-card {
            background: rgba(31, 41, 55, 0.95);
            border-color: #374151;
        }
        html.dark nav {
            background-color: #1f2937 !important;
            color: #f3f4f6;
        }
        html.dark h1, html.dark h2, html.dark h3, html.dark span.text-gray-900 {
            color: #f9fafb !important;
        }
        html.dark p.text-gray-600, html.dark p.text-gray-500, html.dark div.text-gray-500 {
            color: #d1d5db !important;
        }
        html.dark textarea {
            background-color: #374151;
            color: white;
            border-color: #4b5563;
        }
        html.dark .bg-white {
            background-color: #1f2937 !important;
        }
        html.dark .bg-indigo-50 {
            background-color: rgba(107, 70, 193, 0.2) !important;
        }
        html.dark .bg-gray-200 { background-color: #374151 !important; border-color: #1f2937 !important; }
        html.dark .text-gray-800 { color: #f3f4f6 !important; }
        html.dark .text-gray-900 { color: #f9fafb !important; }
        html.dark .border-gray-200, html.dark .border-gray-100 { border-color: #374151 !important; }
"""

js_addition = """
    <script>
        // Dark Mode Logic
        const htmlDoc = document.documentElement;
        const sunBtn = document.getElementById('theme-sun');
        const moonBtn = document.getElementById('theme-moon');

        // Check preference
        if (localStorage.theme === 'dark' || (!('theme' in localStorage) && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
            htmlDoc.classList.add('dark');
        } else {
            htmlDoc.classList.remove('dark');
        }

        if (sunBtn && moonBtn) {
            sunBtn.addEventListener('click', () => { 
                htmlDoc.classList.remove('dark'); 
                localStorage.theme = 'light'; 
            });
            moonBtn.addEventListener('click', () => { 
                htmlDoc.classList.add('dark'); 
                localStorage.theme = 'dark'; 
            });
        }
    </script>
</body>
"""

nav_buttons_old = """            <div class="flex space-x-4 text-gray-500">
                <button class="hover:text-primary"><i class="fas fa-sun text-lg"></i></button>
                <button class="hover:text-primary"><i class="fas fa-moon text-lg"></i></button>
            </div>"""

nav_buttons_new = """            <div class="flex space-x-4 text-gray-500">
                <button id="theme-sun" class="hover:text-primary cursor-pointer transition-colors" title="Light Mode"><i class="fas fa-sun text-lg"></i></button>
                <button id="theme-moon" class="hover:text-primary cursor-pointer transition-colors" title="Dark Mode"><i class="fas fa-moon text-lg"></i></button>
            </div>"""

tailwind_old = "tailwind.config = {"
tailwind_new = "tailwind.config = { darkMode: 'class',"

template_dir = r"c:\Users\akhil\OneDrive\ai ans scoiety\templates"

for filepath in glob.glob(os.path.join(template_dir, "*.html")):
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Add CSS
    if "/* Dark Mode Overrides */" not in content:
        content = content.replace("</style>", css_addition + "</style>")
    
    # Add JS
    if "Dark Mode Logic" not in content:
        content = content.replace("</body>", js_addition)
        
    # Update Nav Buttons
    content = content.replace(nav_buttons_old, nav_buttons_new)
    
    # Update Tailwind config
    if "darkMode: 'class'" not in content:
        content = content.replace(tailwind_old, tailwind_new)
        
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)

print("Dark mode implemented successfully across all templates.")
