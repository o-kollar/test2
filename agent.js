(function() {
    // --- PRE-FLIGHT CHECK: PREVENT DUPLICATE INJECTIONS ---
    if (document.getElementById('m8-app-container')) {
        const container = document.getElementById('m8-app-container');
        container.style.display = container.style.display === 'none' ? 'flex' : 'none';
        console.log('M8 container already exists. Toggled visibility.');
        return;
    }

    // --- DEPENDENCY LOADER ---
    const loadResource = (tag, attrs) => {
        return new Promise((resolve, reject) => {
            const el = document.createElement(tag);
            el.onload = resolve;
            el.onerror = () => reject(new Error(`Failed to load resource: ${attrs.href || attrs.src}`));
            for (const key in attrs) {
                el[key] = attrs[key];
            }
            document.head.appendChild(el);
        });
    };

    // --- MAIN INITIALIZATION FUNCTION ---
    async function initializeM8() {
        console.log("M8: Initializing...");

        // --- A. SHOW LOADING INDICATOR ---
        const loadingIndicator = document.createElement('div');
        loadingIndicator.id = 'm8-loading-indicator';
        loadingIndicator.innerText = 'Loading M8 Agent...';
        Object.assign(loadingIndicator.style, {
            position: 'fixed', top: '20px', right: '20px', padding: '10px 20px',
            backgroundColor: '#1e293b', color: 'white', zIndex: '2147483647',
            borderRadius: '5px', boxShadow: '0 4px 15px rgba(0,0,0,0.3)',
            fontFamily: 'sans-serif', fontSize: '14px', transition: 'background-color 0.3s'
        });
        document.body.appendChild(loadingIndicator);

        try {
            // --- B. LOAD ALL EXTERNAL DEPENDENCIES ---
            await Promise.all([
                loadResource('script', { src: 'https://cdn.tailwindcss.com' }),
                loadResource('script', { src: 'https://cdn.jsdelivr.net/npm/marked/marked.min.js' }),
                loadResource('script', { src: 'https://cdn.jsdelivr.net/npm/chart.js' }),
                loadResource('link', { rel: 'stylesheet', href: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css', crossorigin: 'anonymous' }),
                loadResource('script', { src: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js', defer: true, crossorigin: 'anonymous' }),
                loadResource('script', { src: 'https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js', defer: true, crossorigin: 'anonymous' }),
                loadResource('link', { rel: 'stylesheet', href: 'https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.css', type: 'text/css' }),
                loadResource('script', { src: 'https://cdnjs.cloudflare.com/ajax/libs/vis/4.21.0/vis.min.js', type: 'text/javascript' }),
                loadResource('link', { rel: 'stylesheet', href: 'https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:opsz,wght,FILL,GRAD@20..48,100..700,0..1,-50..200' })
            ]);
            // Load Vue's global build last
            await loadResource('script', { src: 'https://unpkg.com/vue@3/dist/vue.global.js' });

            console.log("M8: All dependencies loaded.");

            // --- C. CREATE THE APP'S FLOATING CONTAINER ---
            const appContainer = document.createElement('div');
            appContainer.id = 'm8-app-container';
            // Set initial class. Vue will manage 'dark' class on this element.
            appContainer.className = "bg-slate-100 text-slate-800 antialiased";
            Object.assign(appContainer.style, {
                position: 'fixed', top: '50px', right: '50px', width: '90%', maxWidth: '800px',
                height: 'calc(100vh - 100px)', maxHeight: '850px', zIndex: '2147483646',
                boxShadow: '0 10px 25px -5px rgba(0,0,0,0.2), 0 8px 10px -6px rgba(0,0,0,0.2)',
                borderRadius: '0.75rem', display: 'flex', flexDirection: 'column',
                overflow: 'hidden', resize: 'both', minWidth: '450px', minHeight: '500px'
            });

            // --- D. INJECT THE APP'S HTML STRUCTURE ---
            const appHTML = `
                <div id="m8-header" style="cursor: move; background-color: #f1f5f9; padding: 8px 12px; border-bottom: 1px solid #e2e8f0; display: flex; align-items: center; justify-content: space-between; user-select: none; flex-shrink: 0;">
                    <span style="font-weight: bold; color: #1e293b;">M8 Agent</span>
                    <div>
                        <button id="m8-hide-btn" title="Minimize M8" style="background: none; border: none; font-size: 24px; line-height: 1; cursor: pointer; color: #64748b; padding: 0 5px;">_</button>
                        <button id="m8-close-btn" title="Close M8" style="background: none; border: none; font-size: 24px; line-height: 1; cursor: pointer; color: #64748b; padding: 0 5px;">×</button>
                    </div>
                </div>
                <div id="m8-content-wrapper" style="flex: 1; min-height: 0; display: flex; flex-direction: column;">
                    <!-- The original app's root element will be placed here -->
                    <div id="app" class="flex flex-col h-screen w-full">
                        <!-- M8 Header with Integrated Tabs -->
                        <header class="flex-shrink-0   backdrop-blur-xl flex items-center justify-between  px-3">
                            <!-- Logo -->
                            <h1 @click="openSettingsModal" class="text-xl font-bold text-zinc-800 dark:text-zinc-200 cursor-pointer hover:opacity-80 transition-opacity flex-shrink-0 pr-4">M8</h1>

                        <!-- Tabs Container -->
                            <div class="flex-1 flex items-center min-w-0">
                                <div class="flex-1 min-w-0 overflow-x-auto" id="tab-container">
                                    <ul class="relative flex p-1 list-none rounded-lg" role="list">
                                        <li v-for="tab in tabs" :key="tab.id" class="z-10 flex-auto text-center">
                                            <a @click="selectTab(tab.id)"
                                            :class="[
                                                'z-10 flex items-center justify-center w-full px-3 py-1.5 text-sm transition-all ease-in-out border-0 rounded-md cursor-pointer whitespace-nowrap',
                                                activeTabId === tab.id
                                                    ? 'bg-white shadow text-slate-800 font-medium dark:bg-slate-700 dark:text-slate-100'
                                                    : 'text-slate-600 dark:text-slate-300 hover:bg-white/60 dark:hover:bg-slate-700/50'
                                            ]"
                                            role="tab" :aria-selected="activeTabId === tab.id">

                                                <!-- Tool Use Indicator -->
                                                <div v-if="tab.activeTools && tab.activeTools.length > 0" class="flex-1 min-w-0 flex justify-center items-center">
                                                    <div :title="tab.activeTools.map((t, i) => `${i+1}. ${t.name} (${t.status})`).join('\\n')">
                                                        <ol class="flex  gap-2 text-xs font-medium text-slate-500 sm:gap-4 dark:text-slate-400">
                                                            <li v-for="(tool, index) in tab.activeTools" :key="tool.name + index" class="flex items-center justify-center gap-1">
                                                                <!-- Success State -->
                                                                <span v-if="tool.status === 'success'" class="flex items-center text-green-600 dark:text-green-400">
                                                                    <span class="rounded-full bg-green-100 dark:bg-green-500/20 p-1.5 flex items-center justify-center">
                                                                        <span class="material-symbols-outlined !text-xs !leading-none">check</span>
                                                                    </span>
                                                                </span>
                                                                <!-- Running State -->
                                                                <span v-else-if="tool.status === 'running'" class="flex items-center justify-center gap-2 text-amber-600 dark:text-blue-400 animate-pulse">
                                                                    <span class="h-6 w-6 rounded-full bg-amber-100 dark:bg-blue-500/20 flex items-center justify-center">
                                                                        <span class="material-symbols-outlined !text-base">{{ getToolIcon(tool.name) }}</span>
                                                                    </span>
                                                                    <span class="font-semibold inline truncate max-w-[100px]">{{ tool.name }}</span>
                                                                </span>
                                                                <!-- Pending State -->
                                                                <span v-else class="flex items-center justify-center gap-2 text-slate-500 dark:text-slate-400">
                                                                    <span class="h-6 w-6 rounded-sm bg-slate-200 dark:bg-slate-600 flex items-center justify-center">
                                                                        <span class="material-symbols-outlined !text-base text-slate-600 dark:text-slate-300">{{ getToolIcon(tool.name) }}</span>
                                                                    </span>
                                                                    <span class="inline truncate max-w-[100px]">{{ tool.name }}</span>
                                                                </span>
                                                            </li>
                                                        </ol>
                                                    </div>
                                                </div>


                                                <!-- Tab name (when no tool is active) -->
                                                <span v-else class="truncate">{{ tab.name }}</span>
                                                
                                                <button @click.stop="closeTab(tab.id)" v-if="tabs.length > 1"
                                                        class="ml-2 -mr-1 p-0.5 rounded-full text-slate-500 hover:bg-slate-400/20 dark:text-slate-400 dark:hover:bg-slate-600/50">
                                                    <span class="material-symbols-outlined !text-sm !leading-none">delete</span>
                                                </button>
                                            </a>
                                        </li>
                                    </ul>
                                </div>
                                <button @click="addNewTab" class="ml-2 p-2 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 flex-shrink-0" title="New Chat">
                                    <span class="material-symbols-outlined !text-lg">add</span>
                                </button>
                            </div>
                            
                            <div class="flex-shrink-0 pl-4"></div>
                        </header>


                        <div id="chat-ui-area" class="flex-1 flex flex-col overflow-hidden bg-white dark:bg-slate-800">
                        <div ref="chatContainerRef" id="chat-container" class="flex-1 p-4 sm:p-6 overflow-y-auto flex flex-col space-y-5">
                            
                            <message-item
                            v-if="activeTab"
                            v-for="msg in activeTab.messages"
                            :key="msg.id"
                            :message="msg"
                            @edit="startEdit"
                            @copy="performCopy"
                            ></message-item>
                            
                        </div>
                        <form @submit.prevent="handleSendMessage" class="p-3 sm:p-4 border-t border-slate-200 dark:border-slate-700 bg-white dark:bg-slate-800">
                            <div class="relative flex items-end space-x-2">
                            <button type="button" @click="triggerFileInput" :disabled="isLoading" class="flex-shrink-0 w-10 h-10 flex items-center justify-center text-slate-500 hover:text-slate-700 dark:text-slate-400 dark:hover:text-slate-200 disabled:opacity-50 transition-colors" title="Attach files">
                                <span class="material-symbols-outlined !text-xl">attach_file</span>
                            </button>
                            <input type="file" ref="fileInputRef" @change="handleFileUpload" multiple class="hidden" />

                            <div class="relative w-full min-w-[200px] flex-grow">
                                <textarea
                                id="-input" rows="1" v-model="newMessageText"
                                @keydown.enter.exact.prevent="handleSendMessage" @input="autoGrowTextarea"
                                :disabled="isLoading"
                                placeholder="Message M8..."
                                class="peer w-full resize-none border-b-2 border-slate-300 dark:border-slate-600 bg-transparent pt-6 pb-2 font-sans text-sm font-normal text-slate-800 dark:text-slate-200 outline outline-0 transition-all placeholder:text-transparent focus:border-zinc-800 dark:focus:border-zinc-300 focus:outline-0 disabled:resize-none disabled:border-0 disabled:bg-white dark:disabled:bg-slate-800 max-h-32"
                                style="line-height: 1.5rem;"
                                ></textarea>
                                <label for="-input"
                                class="after:content[' '] pointer-events-none absolute left-0 -top-2.5 flex h-full w-full select-none !overflow-visible truncate text-[11px] font-normal leading-tight text-slate-500 dark:text-slate-400 transition-all after:absolute after:-bottom-0.5 after:block after:w-full after:scale-x-0 after:border-b-2 after:border-zinc-800 dark:after:border-zinc-300 after:transition-transform after:duration-300 peer-placeholder-shown:text-sm peer-placeholder-shown:leading-[4.2] peer-placeholder-shown:text-slate-500 peer-focus:text-[11px] peer-focus:leading-tight peer-focus:text-zinc-800 dark:peer-focus:text-zinc-300 peer-focus:after:scale-x-100 peer-focus:after:border-zinc-800 dark:peer-focus:after:border-zinc-300 peer-disabled:text-transparent peer-disabled:peer-placeholder-shown:text-slate-500">
                                Message M8...
                                </label>
                            </div>
                            <button type="submit" :disabled="isLoading || !newMessageText.trim() && !editingMessage" class="flex-shrink-0 w-10 h-10 flex items-center justify-center text-white bg-zinc-800 rounded-full hover:bg-zinc-900 focus:ring-2 focus:outline-none focus:ring-zinc-500 disabled:bg-zinc-800/60 disabled:cursor-not-allowed transition-colors dark:bg-zinc-300 dark:text-zinc-900 dark:hover:bg-zinc-200 dark:disabled:bg-zinc-300/60" title="Send message">
                                <span v-if="isLoading"><span class="material-symbols-outlined animate-spin !text-xl">progress_activity</span></span>
                                <span v-else><span class="material-symbols-outlined !text-xl">send</span></span>
                            </button>
                            </div>
                        </form>
                        </div>

                        <!-- Settings Modal -->
                        <div v-if="showSettingsModal" @click.self="closeSettingsModal" class="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4 transition-opacity">
                        <div class="bg-slate-50 dark:bg-slate-800 rounded-lg shadow-xl w-full max-w-md p-6 relative">
                            <button @click="closeSettingsModal" class="absolute top-2 right-2 text-slate-500 hover:text-slate-800 dark:text-slate-400 dark:hover:text-slate-100 text-2xl leading-none font-bold p-2">×</button>
                            <h2 class="text-lg font-bold text-zinc-800 dark:text-zinc-200 mb-4 border-b border-slate-300 dark:border-slate-600 pb-2">Settings</h2>
                            <form @submit.prevent="saveSettings">
                            <div class="space-y-4">
                                <div>
                                <label for="model-select" class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">AI Model</label>
                                <select id="model-select" v-model="settingsModel" class="w-full p-2 border border-slate-300 rounded-md focus:ring-zinc-500 focus:border-zinc-500 bg-white dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200">
                                    <option value="gemini-2.5-flash-lite">Gemini 2.5 Flash Lite</option>
                                    <option value="gemini-2.5-flash">Gemini 2.5 Flash</option>
                                    <option value="gemini-2.5-pro">Gemini 2.5 Pro</option>
                                </select>
                                </div>
                                <div>
                                <label for="api-key-input" class="block text-sm font-medium text-slate-700 dark:text-slate-300 mb-1">Gemini API Key</label>
                                <input type="password" id="api-key-input" v-model="settingsApiKey" placeholder="Enter your Gemini API Key" class="w-full p-2 border border-slate-300 rounded-md focus:ring-zinc-500 focus:border-zinc-500 bg-white dark:bg-slate-700 dark:border-slate-600 dark:text-slate-200">
                                <p class="text-xs text-slate-500 dark:text-slate-400 mt-1">Get a key from Google AI Studio. It's stored in your browser's local storage.</p>
                                </div>
                            </div>
                            <div class="mt-6 flex justify-end">
                                <button type="submit" class="px-4 py-2 bg-zinc-800 text-white rounded-md hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-500 dark:bg-zinc-700 dark:hover:bg-zinc-600">Save & Close</button>
                            </div>
                            </form>
                        </div>
                        </div>

                    </div>
                </div>
            `;
            appContainer.innerHTML = appHTML;
            document.body.appendChild(appContainer);

            // --- E. INJECT THE SCOPED CSS ---
            const style = document.createElement('style');
            style.textContent = `
                /* Scope all styles to the container to avoid conflicts with the host page */
                #m8-app-container.dark #m8-header { background-color: #1e293b; border-bottom-color: #334155; }
                #m8-app-container.dark #m8-header span { color: #f1f5f9; }
                #m8-app-container.dark #m8-header button { color: #94a3b8; }
                
                #m8-app-container @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
                #m8-app-container .animate-blink { animation: blink 1s step-end infinite; }
                #m8-app-container .rendered-content { display: inline-block; width: 100%; }
                #m8-app-container .typing-cursor { display: inline-block; width: 2px; height: 1.2em; background-color: currentcolor; vertical-align: text-bottom; margin-left: 1px; }
                #m8-app-container .katex-display > .katex { text-align: initial; overflow-x: auto; overflow-y: hidden; max-width: 100%; }
                #m8-app-container #chat-container::-webkit-scrollbar { width: 6px; }
                #m8-app-container #chat-container::-webkit-scrollbar-track { background: transparent; }
                #m8-app-container #chat-container::-webkit-scrollbar-thumb { background-color: rgba(0,0,0,0.1); border-radius: 10px; }
                #m8-app-container .dark #chat-container::-webkit-scrollbar-thumb { background-color: rgba(255,255,255,0.1); }
                #m8-app-container #chat-container::-webkit-scrollbar-thumb:hover { background-color: rgba(0,0,0,0.2); }
                #m8-app-container .dark #chat-container::-webkit-scrollbar-thumb:hover { background-color: rgba(255,255,255,0.2); }
                #m8-app-container #tab-container::-webkit-scrollbar { height: 4px; }
                #m8-app-container #tab-container::-webkit-scrollbar-track { background: transparent; }
                #m8-app-container #tab-container::-webkit-scrollbar-thumb { background-color: rgba(0,0,0,0.1); border-radius: 10px; }
                #m8-app-container .dark #tab-container::-webkit-scrollbar-thumb { background-color: rgba(255,255,255,0.2); }
                #m8-app-container .graph-host-container, #m8-app-container .chart-host-container {
                    position: relative; height: 350px; width: 100%; border-radius: 0.5rem;
                    border: 1px solid #e2e8f0; margin-top: 1rem; margin-bottom: 1rem;
                    transition: opacity 0.5s ease-in-out, max-height 0.5s ease-in-out; overflow: hidden;
                }
                #m8-app-container .chart-host-container { padding: 1rem; }
                #m8-app-container .dark .graph-host-container, #m8-app-container .dark .chart-host-container { border-color: #334155; }
                #m8-app-container .node-details-card {
                    position: absolute; top: 10px; right: 10px; width: 280px; max-height: calc(100% - 20px);
                    background-color: #fff; border: 1px solid #e2e8f0; border-radius: 0.375rem;
                    padding: 1rem; z-index: 100; display: flex; flex-direction: column;
                }
                #m8-app-container .dark .node-details-card { background-color: #334155; border-color: #475569; color: #f1f5f9; }
                #m8-app-container .node-details-card h3 { margin-top: 0; margin-bottom: 0.75rem; font-size: 0.875rem; font-weight: 600; border-bottom: 1px solid #e2e8f0; padding-bottom: 0.5rem; color: #1e293b; }
                #m8-app-container .dark .node-details-card h3 { border-bottom-color: #475569; color: #f1f5f9; }
                #m8-app-container .node-details-card-body { flex-grow: 1; overflow-y: auto; font-size: 0.875rem; line-height: 1.4; }
                #m8-app-container .node-details-card-body::-webkit-scrollbar { width: 5px; }
                #m8-app-container .node-details-card-body::-webkit-scrollbar-track { background: transparent; }
                #m8-app-container .node-details-card-body::-webkit-scrollbar-thumb { background-color: rgba(0,0,0,0.2); border-radius: 10px; }
                #m8-app-container .dark .node-details-card-body::-webkit-scrollbar-thumb { background-color: rgba(255,255,255,0.2); }
                #m8-app-container .node-details-card button.close-btn { position: absolute; top: 0.5rem; right: 0.5rem; background: transparent; border: none; font-size: 1.5rem; line-height: 1; cursor: pointer; color: #64748b; padding: 0.25rem; }
                #m8-app-container .dark .node-details-card button.close-btn { color: #94a3b8; }
                #m8-app-container .node-details-card button.close-btn:hover { color: #1e293b; }
                #m8-app-container .dark .node-details-card button.close-btn:hover { color: #f1f5f9; }
            `;
            document.head.appendChild(style);

            // --- F. CONFIGURE TAILWIND AND INJECT THE VUE APP LOGIC ---
            // Configure Tailwind to operate in 'class' mode and apply styles within our container
            window.tailwind.config = {
                darkMode: 'class',
                // Important: Tell Tailwind where to look for classes
                content: ["#m8-app-container"],
            };
            
            const appScript = document.createElement('script');
            const originalScriptContent = `
            // --- The original script content from the user's file ---

            // This is a direct copy of the user's script, with three modifications noted below.

            const { createApp, ref, nextTick, watch, onMounted, onUnmounted, computed } = Vue; // MODIFICATION 1: Use global Vue object
    
            // --- Configure Marked for link handling ---
            const renderer = new marked.Renderer();
            renderer.link = (href, title, text) => {
              // Ensure links open in a new tab
              return \`<a href="\${href}" target="_blank" rel="noopener noreferrer" title="\${title || ''}">\${text}</a>\`;
            };
            marked.setOptions({
              renderer: renderer,
              gfm: true,        // Use GitHub Flavored Markdown (for autolinking)
              breaks: true,     // Convert single line breaks to <br>
              pedantic: false,
              sanitize: false   // Be aware of security implications if rendering user-provided content
            });

            // --- MessageItem Component ---
            const MessageItem = {
                props: ['message'], emits: ['edit', 'copy'],
                setup(props, { emit }) {
                    const contentRef = ref(null), parsedContent = ref(''), isUser = props.message.role === 'user';
                    const graphHostElement = ref(null);
                    const chartCanvasRef = ref(null);
                    const chartInstance = ref(null);
                    const isDarkMode = ref(visGraphManager.isDarkMode());
                    const selectedRawNode = ref(null);
                    const handleNodeSelectionChange = (nodeData) => { selectedRawNode.value = nodeData; };
                    const displayedNodeDetails = computed(() => { if (selectedRawNode.value && selectedRawNode.value.body && selectedRawNode.value.body.trim() !== '') { return { id: selectedRawNode.value.label, parsedBody: marked.parse(selectedRawNode.value.body) }; } return null; });
                    const closeNodeCard = () => { visGraphManager.setSelectedNode(null); };
                    const renderChart = () => { if (chartInstance.value) { chartInstance.value.destroy(); chartInstance.value = null; } if (!chartCanvasRef.value || !props.message.chartConfig) return; const { type, data, options } = props.message.chartConfig; const themeIsDark = document.getElementById('m8-app-container')?.classList.contains('dark'); const tickColor = themeIsDark ? 'rgba(255, 255, 255, 0.2)' : 'rgba(0, 0, 0, 0.1)'; const fontColor = themeIsDark ? '#cbd5e1' : '#475569'; const defaultOptions = { responsive: true, maintainAspectRatio: false, plugins: { legend: { labels: { color: fontColor } }, title: { display: !!(options && options.title), text: options ? options.title : '', color: fontColor, font: { size: 16 } } }, scales: { y: { ticks: { color: fontColor }, grid: { color: tickColor } }, x: { ticks: { color: fontColor }, grid: { color: tickColor } } } }; if (type === 'pie' || type === 'doughnut' || type === 'polarArea' || type === 'radar') { delete defaultOptions.scales; if (type === 'radar') { defaultOptions.scales = { r: { angleLines: { color: tickColor }, grid: { color: tickColor }, pointLabels: { color: fontColor, font: { size: 12 } }, ticks: { color: fontColor, backdropColor: 'transparent' } } }; } } const config = { type: type, data: data, options: defaultOptions }; chartInstance.value = new Chart(chartCanvasRef.value, config); };
                    const updateContentAndRenderMath = (text) => {
                      if (props.message.isGraphDisplaySlot || props.message.isChartDisplaySlot) { parsedContent.value = ''; return; }
                      let html;
                      if (props.message.isRawHtml) { html = text; } 
                      else if (props.message.isError) { html = text; } 
                      else if (props.message.role === 'model' || props.message.isSystemToolResponse || props.message.isCodeDisplay) { html = marked.parse(text || (props.message.isStreaming ? ' ' : '')); } 
                      else { html = marked.parse(text || ''); }

                      if (props.message.isCodeDisplay) { html = \`<div class="p-2.5 my-2 text-slate-800 dark:text-slate-200 bg-slate-200/50 dark:bg-slate-700/50 rounded-lg border border-slate-300 dark:border-slate-600/50">\${html}</div>\`; }
                      else if (props.message.isSystemToolResponse) { html = \`<div class="p-2.5 text-xs italic bg-slate-200 dark:bg-slate-700/60 rounded-lg text-slate-500 dark:text-slate-400 border border-slate-300 dark:border-slate-600/50">\${html}</div>\`; }
                      else if (props.message.isError) { html = \`<div class="p-2.5 border border-red-400/50 rounded-lg bg-red-100/70 text-red-700 dark:bg-red-900/30 dark:text-red-300 dark:border-red-500/50">\${html}</div>\`; }
                      else if (isUser) { html = \`<div class="text-slate-700 dark:text-slate-200 text-right">\${html}</div>\`; }
                      else { html = \`<div class="text-slate-800 dark:text-slate-200 text-left">\${html}</div>\`; }

                      parsedContent.value = html;
                      nextTick(() => { if (contentRef.value && !props.message.isError && !props.message.isRawHtml && (props.message.role === 'model' || (text && text.includes('$')))) { try { if (typeof renderMathInElement === 'function') renderMathInElement(contentRef.value, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false }); } catch (e) { console.warn("KaTeX:", e); } } });
                    };
                    watch(() => props.message.parts[0].text, (nt) => { if (!props.message.isGraphDisplaySlot && !props.message.isChartDisplaySlot) updateContentAndRenderMath(nt); }, { immediate: true });
                    watch(() => props.message.isStreaming, (is, was) => { if (was && !is && !props.message.isGraphDisplaySlot && !props.message.isChartDisplaySlot) updateContentAndRenderMath(props.message.parts[0].text); });
                    watch(selectedRawNode, (newNodeData) => { if (newNodeData && newNodeData.body && graphHostElement.value) { nextTick(() => { const cardBody = graphHostElement.value.querySelector('.node-details-card-body'); if (cardBody && typeof renderMathInElement === 'function' && newNodeData.body.includes('$')) { try { renderMathInElement(cardBody, { delimiters: [{ left: '$$', right: '$$', display: true }, { left: '$', right: '$', display: false }], throwOnError: false }); } catch (e) { console.warn("KaTeX in card:", e); } } }); } }, { deep: true });
                    const handleEdit = () => emit('edit', props.message), handleCopy = () => emit('copy', props.message);
                    onMounted(() => {
                        isDarkMode.value = visGraphManager.isDarkMode();
                        let themeObserver = null;
                        if (props.message.isGraphDisplaySlot && graphHostElement.value) { visGraphManager.ensureInitialized(graphHostElement.value, handleNodeSelectionChange); }
                        if (props.message.isChartDisplaySlot) {
                            renderChart();
                            themeObserver = new MutationObserver((mutations) => {
                                for (const mutation of mutations) { if (mutation.attributeName === 'class') { isDarkMode.value = visGraphManager.isDarkMode(); renderChart(); } }
                            });
                            const container = document.getElementById('m8-app-container');
                            if(container) themeObserver.observe(container, { attributes: true });
                        }
                        onUnmounted(() => { if (themeObserver) themeObserver.disconnect(); });
                    });
                    onUnmounted(() => {
                        if (props.message.isGraphDisplaySlot && visGraphManager.activeContainerElement === graphHostElement.value) { visGraphManager.destroyVisualization(); selectedRawNode.value = null; }
                        if (chartInstance.value) chartInstance.value.destroy();
                    });
                    return { isUser, handleEdit, handleCopy, contentRef, parsedContent, message: props.message, graphHostElement, isDarkMode, displayedNodeDetails, closeNodeCard, chartCanvasRef };
                },
                template: \`
                <div v-if="message.isGraphDisplaySlot" class="w-full my-3">
                    <div ref="graphHostElement" class="graph-host-container bg-white/50 dark:bg-slate-800/50">
                        <div v-if="displayedNodeDetails" class="node-details-card" :class="{ 'dark': isDarkMode }">
                            <h3>{{ displayedNodeDetails.id }}</h3>
                            <div class="node-details-card-body prose prose-sm dark:prose-invert max-w-none" v-html="displayedNodeDetails.parsedBody"></div>
                            <button @click="closeNodeCard" class="close-btn" title="Close entry details">×</button>
                        </div>
                    </div>
                </div>
                <div v-else-if="message.isChartDisplaySlot" class="w-full my-3">
                    <div class="chart-host-container bg-white/50 dark:bg-slate-800/50">
                        <canvas ref="chartCanvasRef"></canvas>
                    </div>
                </div>
                <div v-else :class="['w-full flex group relative', isUser ? 'justify-end' : 'justify-start']" :data-message-id="message.id">
                  <div :class="['max-w-[85%] sm:max-w-[80%']">
                      <div ref="contentRef" class="rendered-content prose prose-sm max-w-none prose-p:my-0.5 prose-ul:my-1 prose-ol:my-1 prose-li:my-0 prose-headings:my-1 prose-pre:my-1.5 prose-a:text-blue-600 hover:prose-a:underline dark:prose-a:text-blue-400" v-html="parsedContent"></div>
                      <span v-if="message.isStreaming && !message.isError && !message.isSystemToolResponse" :class="['typing-cursor animate-blink mt-0.5', isUser ? 'float-right mr-1' : 'float-left ml-1']"></span>
                  </div>
                  <div v-if="!message.isError && !message.isSystemToolResponse && !message.isGraphDisplaySlot && !message.isChartDisplaySlot && !message.isCodeDisplay && (isUser || (!message.isStreaming && message.role === 'model'))" :class="['absolute top-1/2 -translate-y-1/2 flex items-center space-x-1.5 z-10 opacity-0 group-hover:opacity-100 transition-opacity duration-150', isUser ? 'left-0 -translate-x-full pr-2' : 'right-0 translate-x-full pl-2']">
                      <button v-if="isUser" @click="handleEdit" title="Edit" class="p-1.5 rounded-full bg-slate-100/80 hover:bg-slate-200 text-slate-500 border border-slate-300/70 focus:outline-none focus:ring-1 focus:ring-zinc-500 dark:bg-slate-700 dark:hover:bg-slate-600 dark:text-slate-400 dark:border-slate-600"><span class="material-symbols-outlined !text-sm !leading-none">edit</span></button>
                      <button @click="handleCopy" title="Copy" class="copy-button p-1.5 rounded-full bg-slate-100/80 hover:bg-slate-200 text-slate-500 border border-slate-300/70 focus:outline-none focus:ring-1 focus:ring-zinc-500 dark:bg-slate-700 dark:hover:bg-slate-600 dark:text-slate-400 dark:border-slate-600"><span class="material-symbols-outlined !text-sm !leading-none">content_copy</span></button>
                  </div>
                </div>
              \`
            };
            
            // --- vis.js Graph Manager (with sync callback fixes) ---
            const visGraphManager = {
                network: null, nodes: new vis.DataSet(), edges: new vis.DataSet(), activeContainerElement: null,
                selectionCallback: null, isDarkMode: () => document.getElementById('m8-app-container')?.classList.contains('dark'), // MODIFICATION 2: Check our container
                getOptions(isDark) {
                    const fontColor = isDark ? '#e2e8f0' : '#1e293b'; const nodeBg = isDark ? '#334155' : '#f8fafc'; const nodeBorder = isDark ? '#64748b' : '#cbd5e1'; const edgeColor = isDark ? '#64748b' : '#94a3b8';
                    return { autoResize: true, height: '100%', width: '100%', nodes: { shape: 'box', borderWidth: 1.5, color: { background: nodeBg, border: nodeBorder, highlight: { background: isDark ? '#475569' : '#e2e8f0', border: isDark ? '#94a3b8' : '#64748b' } }, font: { color: fontColor, size: 14, face: 'system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif' }, margin: { top: 10, right: 15, bottom: 10, left: 15 }, widthConstraint: { minimum: 100, maximum: 200 }, shapeProperties: { borderRadius: 4 } }, edges: { color: { color: edgeColor, highlight: isDark ? '#cbd5e1' : '#475569' }, arrows: { to: { enabled: false } }, smooth: { enabled: true, type: 'dynamic' } }, physics: { enabled: true, barnesHut: { gravitationalConstant: -10000, centralGravity: 0.1, springLength: 150, springConstant: 0.05, damping: 0.1, avoidOverlap: 0.2 }, solver: 'barnesHut', stabilization: { iterations: 1000, fit: true } }, interaction: { dragNodes: true, dragView: true, zoomView: true, hover: true, tooltipDelay: 200, navigationButtons: false } };
                },
                loadTabData(graphData) { this.nodes.clear(); this.edges.clear(); if (graphData && graphData.nodes) this.nodes.add(graphData.nodes); if (graphData && graphData.edges) this.edges.add(graphData.edges); if (this.network) this.network.setData({ nodes: this.nodes, edges: this.edges }); },
                getCurrentGraphData() { if (this.network) this.network.storePositions(); return { nodes: this.nodes.get({ fields: ['id', 'label', 'body', 'x', 'y', 'title', 'type'] }), edges: this.edges.get({ fields: ['from', 'to', 'id'] }) }; },
                ensureInitialized(el, cb) { if (!el || (this.network && this.activeContainerElement === el)) return; this.destroyVisualization(); this.activeContainerElement = el; this.selectionCallback = cb; const d = { nodes: this.nodes, edges: this.edges }; const o = this.getOptions(this.isDarkMode()); this.network = new vis.Network(el, d, o); this.network.on('click', (p) => { const nId = p.nodes.length > 0 ? p.nodes[0] : null; if (this.selectionCallback) this.selectionCallback(nId ? this.nodes.get(nId) : null); }); this.network.on('dragEnd', () => this.syncCallback && this.syncCallback()); this.network.on('stabilizationIterationsDone', () => { if (this.network) this.network.storePositions(); this.syncCallback && this.syncCallback(); }); },
                setSyncCallback(callback) { this.syncCallback = callback; },
                destroyVisualization() { if (this.network) { this.network.destroy(); this.network = null; } this.activeContainerElement = null; this.selectionCallback = null; },
                updateTheme() { if (this.network) this.network.setOptions(this.getOptions(this.isDarkMode())); },
                zoomToFitAllNodes(duration = 750) { if (this.network && this.nodes.length > 0) this.network.fit({ animation: { duration, easingFunction: 'easeInOutQuad' } }); },
                resetZoom(duration = 750) { if (this.network) this.network.moveTo({ scale: 1.0, animation: { duration, easingFunction: 'easeInOutQuad' } }); },
                setSelectedNode(nodeId) { if (!this.network) return; if (nodeId) { this.network.selectNodes([nodeId]); this.network.focus(nodeId, { scale: 1.2, animation: { duration: 500 } }); } else { this.network.unselectAll(); this.network.fit({ animation: { duration: 500 } }); } },
                addNode(id, body = "", extraData = {}) { if (!id || typeof id !== 'string' || id.trim() === "") return { success: false, message: "Key (ID) is required." }; id = id.trim(); if (this.nodes.get(id)) return { success: false, message: \`Entry "\${id}" already exists.\` }; const nodeData = { id, label: id, body: body || "", title: body || " ", ...extraData }; this.nodes.add(nodeData); this.zoomToFitAllNodes(); this.syncCallback && this.syncCallback(); return { success: true, message: \`Entry "\${id}" created.\` }; },
                editNode(currentId, newIdParam, newBodyParam) { currentId = currentId ? currentId.trim() : null; const newId = newIdParam ? newIdParam.trim() : null; if (!currentId) return { success: false, message: "Current key is required." }; const n = this.nodes.get(currentId); if (!n) return { success: false, message: \`Entry "\${currentId}" not found.\` }; const idChanged = newId && newId !== currentId; if (idChanged && this.nodes.get(newId)) return { success: false, message: \`Cannot rename to "\${newId}": key exists.\` }; let bodyChanged = typeof newBodyParam === 'string' && n.body !== newBodyParam; if (!idChanged && !bodyChanged) return { success: true, message: \`Entry "\${currentId}" was not changed.\` }; if (idChanged) { const nn = { ...n, id: newId, label: newId, body: bodyChanged ? newBodyParam : n.body, title: bodyChanged ? newBodyParam : (n.body || " ") }; delete nn.x; delete nn.y; const cE = this.edges.get({ filter: e => e.from === currentId || e.to === currentId }); const nE = cE.map(e => ({ ...e, from: e.from === currentId ? newId : e.from, to: e.to === currentId ? newId : e.to })); this.nodes.remove(currentId); this.edges.remove(cE.map(e => e.id)); this.nodes.add(nn); this.edges.add(nE); } else { this.nodes.update({ id: currentId, body: newBodyParam, title: newBodyParam || " " }); } const sN = this.network ? this.network.getSelectedNodes() : []; if (sN.length > 0 && sN[0] === currentId) { const fId = idChanged ? newId : currentId; this.setSelectedNode(fId); if (this.selectionCallback) this.selectionCallback(this.nodes.get(fId)); } let msg = \`Entry "\${currentId}" updated.\`; if (idChanged) msg += \` Renamed to "\${newId}".\`; if (bodyChanged) msg += \` Value updated.\`; this.syncCallback && this.syncCallback(); return { success: true, message: msg }; },
                addLink(srcId, tgtId) { if (!srcId || !tgtId) return { success: false, message: "Source and target keys required." }; srcId = srcId.trim(); tgtId = tgtId.trim(); if (srcId === tgtId) return { success: false, message: "Cannot link to self." }; if (!this.nodes.get(srcId)) return { success: false, message: \`Source "\${srcId}" not found.\` }; if (!this.nodes.get(tgtId)) return { success: false, message: \`Target "\${tgtId}" not found.\` }; if (this.edges.get({ filter: e => (e.from === srcId && e.to === tgtId) || (e.from === tgtId && e.to === srcId) }).length > 0) { return { success: false, message: "Link already exists." }; } this.edges.add({ from: srcId, to: tgtId }); this.syncCallback && this.syncCallback(); return { success: true, message: \`Linked "\${srcId}" to "\${tgtId}".\` }; },
                deleteNode(id) { id = id.trim(); if (!this.nodes.get(id)) return { success: false, message: \`Entry "\${id}" not found.\` }; const sN = this.network ? this.network.getSelectedNodes() : []; if (sN.length > 0 && sN[0] === id) this.selectionCallback(null); this.nodes.remove(id); this.syncCallback && this.syncCallback(); return { success: true, message: \`Entry "\${id}" deleted.\` }; },
                highlightSearch(q) { const nq = q ? q.trim().toLowerCase() : ""; const dC = this.getOptions(this.isDarkMode()).nodes.color; const u = this.nodes.map(n => ({ id: n.id, color: (nq && (n.label.toLowerCase().includes(nq) || (n.body && n.body.toLowerCase().includes(nq)))) ? dC.highlight : { background: dC.background, border: dC.border } })); if (u.length > 0) this.nodes.update(u); return { success: true, message: nq ? \`Highlighting: "\${q}".\` : "Highlighting cleared." }; },
                clearGraphData() { this.nodes.clear(); this.edges.clear(); if (this.selectionCallback) this.selectionCallback(null); this.syncCallback && this.syncCallback(); return { success: true, message: "Data store for this tab cleared." }; }
            };
            
            createApp({
              components: { 'message-item': MessageItem },
              setup() {
                // --- App State (Multi-Tab & Global) ---
                const tabs = ref([]);
                const activeTabId = ref(null);
                const newMessageText = ref(''), editingMessage = ref(null);
                const isLoading = ref(false), chatContainerRef = ref(null), fileInputRef = ref(null);
                const activeTempGraphSlotId = ref(null);

                const LS_SESSIONS_KEY = 'm8_chat_sessions_v6';
                const LS_GRAPH_DATA_KEY = 'm8_graph_data_v1';

                const activeTab = computed(() => tabs.value.find(t => t.id === activeTabId.value));
                
                // --- Settings State ---
                const apiKey = ref("");
                const modelName = ref("gemini-2.5-flash-lite");
                const apiUrl = computed(() => \`https://generativelanguage.googleapis.com/v1beta/models/\${modelName.value}:generateContent?key=\${apiKey.value}\`);
                const showSettingsModal = ref(false);
                const settingsModel = ref('');
                const settingsApiKey = ref('');
                const LS_API_KEY_KEY = 'm8_gemini_api_key_v2';
                const LS_MODEL_NAME_KEY = 'm8_gemini_model_name_v2';
                
                // --- Tool Icon Mapping ---
                const toolIconMap = {
                  get_weather_forecast: 'thermometer',
                  web_search: 'search',
                  fetch_page_content: 'web_traffic',
                  store_data: 'database',
                  update_data: 'database',
                  link_data: 'database',
                  delete_data: 'database',
                  search_data: 'database',
                  clear_store: 'database',
                  list_data: 'database',
                  list_files: 'folder',
                  read_file: 'folder',
                  delete_file: 'folder',
                  create_download_link: 'folder',
                  execute_javascript: 'code_blocks',
                  display_chart: 'bar_chart',
                };
                const getToolIcon = (toolName) => {
                  return toolIconMap[toolName] || 'construction'; // Fallback icon
                };

                const saveTabsToLocalStorage = () => { try { localStorage.setItem(LS_SESSIONS_KEY, JSON.stringify(tabs.value)); } catch (e) { console.error("Error saving sessions:", e); } };
                const saveGraphToLocalStorage = () => { try { localStorage.setItem(LS_GRAPH_DATA_KEY, JSON.stringify(visGraphManager.getCurrentGraphData())); } catch (e) { console.error("Error saving graph data:", e); } };
                watch(tabs, saveTabsToLocalStorage, { deep: true });
                watch(activeTabId, (newId) => { if (newId) { scrollToBottom(true); } });
                
                // --- Settings & Tab Management ---
                const openSettingsModal = () => { settingsModel.value = modelName.value; settingsApiKey.value = apiKey.value; showSettingsModal.value = true; };
                const closeSettingsModal = () => { showSettingsModal.value = false; };
                const saveSettings = () => { if (!settingsApiKey.value || settingsApiKey.value.trim() === '') { alert("API Key cannot be empty."); return; } localStorage.setItem(LS_API_KEY_KEY, settingsApiKey.value); localStorage.setItem(LS_MODEL_NAME_KEY, settingsModel.value); apiKey.value = settingsApiKey.value; modelName.value = settingsModel.value; closeSettingsModal(); addOrUpdateMessage('system', \`Settings updated. Using model: <b>\${settingsModel.value}</b>.\`, null, false, false, \`Settings updated.\`, true, false, false, true); };
                const createNewTab = (messages = []) => { const newTabId = Date.now().toString(); const newTab = { id: newTabId, name: \`Chat \${tabs.value.length + 1}\`, messages, activeTools: [] }; tabs.value.push(newTab); return newTabId; };
                const addNewTab = () => { const newTabId = createNewTab([{ id: Date.now().toString(36), role: 'model', parts: [{ text: '👋' }], originalMarkdown: '👋', isStreaming: false, isError: false, isSystemToolResponse: false, isGraphDisplaySlot: false, isCodeDisplay: false, isRawHtml: false }]); selectTab(newTabId); };
                const selectTab = (tabId) => { if (activeTabId.value !== tabId) { activeTabId.value = tabId; } };
                const closeTab = (tabId) => { const index = tabs.value.findIndex(t => t.id === tabId); if (index === -1) return; let newActiveId = null; if (activeTabId.value === tabId) { if (tabs.value.length > 1) { newActiveId = index > 0 ? tabs.value[index - 1].id : tabs.value[1].id; } } tabs.value.splice(index, 1); if (tabs.value.length === 0) { addNewTab(); } else if (newActiveId) { selectTab(newActiveId); } };

                // --- Core Logic & Tool Definitions ---
                const SYSTEM_PROMPT = \`You are M8, an advanced multi-step reasoning assistant with full access to external tools. You combine thoughtful, step-by-step analysis with precise tool use to retrieve, process, and present information.

## How You Work
1. **Plan**: Break down requests into subtasks. Determine logical order and identify needed tools.
2. **Use Tools Thoughtfully**: Call tools for real-time or specialized data. Analyze results before proceeding.
3. **Reason in Steps**: Use multi-step reasoning.
4. **Present Results**: Share clean, concise summaries. Show lists clearly, answer questions directly.

## Interaction Style
- Match the user's tone
- Be conversational, but always grounded and factual, give concise answer unless asked otherwise
- Ask clarifying questions when needed.

# Tools
Your tools are for data management, information retrieval, and code execution.

## Weather
- Use 'get_weather_forecast' with a city name for a general summary.
- To get a detailed hourly forecast for a specific day (e.g., today, tomorrow), provide the 'date' parameter in YYYY-MM-DD format. The current date is provided in the context below.

## Web Search & Page Reading
Use the web tool to access up-to-date information from the web or when responding to the user requires information about their location. Some examples of when to use the web tool include:

- Local Information: Use the web tool to respond to questions that require information about the user's location, such as the weather, local businesses, or events.
- Freshness: If up-to-date information on a topic could potentially change or enhance the answer, call the web tool any time you would otherwise refuse to answer a question because your knowledge might be out of date.
- Niche Information: If the answer would benefit from detailed information not widely known or understood (which might be found on the internet), such as details about a small neighborhood, a less well-known company, or arcane regulations, use web sources directly rather than relying on the distilled knowledge from pretraining.
- Accuracy: If the cost of a small mistake or outdated information is high (e.g., using an outdated version of a software library or not knowing the date of the next game for a sports team), then use the web tool.
- To research a topic on the web, use a two-step process:
- **Step 1:** Use 'web_search' with a query.
- **Step 2:** Review the search results (titles, snippets, and URLs).
- **Step 3:** If a result seems promising, use 'fetch_page_content' with its URL to read the full text.
- **Step 4:** You can then summarize or analyze this text for the user.

## Unified Data & File Store (Graph)
- You have a single, unified data store that works like a graph database. It can hold both regular data entries and files. Everything is shared across all chat tabs.
- **Data Entries:** These are key-value pairs (nodes). Use tools like \\\`store_data\\\`, \\\`update_data\\\`, and \\\`link_data\\\` to manage them.
- **Files:** Files are also nodes in the graph, but with a special type. You can upload them using the "Attach files" button. To manage them, use file-specific tools:
  - \\\`list_files\\\`: See all available files.
  - \\\`read_file\\\`: Read the content of a file.
  - \\\`delete_file\\\`: Remove a file.
  - \\\`create_download_link\\\`: Let the user download a file.
- You can chain tool calls. For example, to store data from a web page, first read the page, then call 'store_data'.

## Chart Display
- To visualize data, use the \\\`display_chart\\\` tool.
- You must specify a \\\`type\\\` (e.g., 'bar', 'line', 'pie', 'doughnut', 'radar') and the \\\`data\\\` object.
- The \\\`data\\\` object needs \\\`labels\\\` (an array of strings) and \\\`datasets\\\` (an array of objects).
- Each dataset needs a \\\`label\\\` (string) and \\\`data\\\` (an array of numbers).

## JavaScript Execution with File I/O
- For complex data processing or custom logic, use the \\\`execute_javascript\\\` tool.
- The code runs in a sandbox but can access the unified store's files via a special \\\`M8\\\` object.
- **You MUST use a \\\`return\\\` statement** to send a value back from your code.
- **File System API within \\\`execute_javascript\\\`:**
  - \\\`M8.files.read(filename: string): string | undefined\\\`: Reads a file node's content.
  - \\\`M8.files.write(filename: string, content: string): void\\\`: Creates or overwrites a file node in the store.
  - \\\`M8.files.list(): string[]\\\`: Lists all available filenames.
- **Example:** To process an uploaded 'data.csv' and save a result.
  - Call \\\`execute_javascript\\\` with the code:
    \\\`const csvText = M8.files.read('data.csv'); if (!csvText) { return "File not found."; } const lines = csvText.split('\\\\n'); const processed = lines.map(l => l.toUpperCase()); M8.files.write('output.txt', processed.join('\\\\n')); return "Processed 'data.csv' and saved to 'output.txt'.";\\\`
 ## rules
 - reply to the user after you have ran all the planned tools
 - structure the answer in well formated markdown
 - when working with numbers use javascript for calculation and charts for visualization\`;
                const geminiTools = [{ functionDeclarations: [ { name: "get_weather_forecast", description: "Fetches the weather forecast. Provides a general summary by default, or a detailed hourly forecast for a specific date if provided.", parameters: { type: "OBJECT", properties: { location: { type: "STRING", description: "The city or location to get the weather for." }, date: { type: "STRING", description: "Optional. A specific date for an hourly forecast, in YYYY-MM-DD format." } }, required: ["location"] } }, { name:"store_data", description:"Stores a new data entry (a key-value pair, not a file) in the unified graph store.", parameters:{ type:"OBJECT", properties:{ key:{type:"STRING",description:"The unique key for the new data entry."}, value:{type:"STRING",description:"Optional. The text/Markdown value for the entry."} }, required:["key"] } }, { name:"update_data", description:"Modifies an existing data entry in the graph store. Cannot modify files.", parameters: { type: "OBJECT", properties: { key: {type: "STRING", description: "Current key of the data entry to edit."}, new_key: {type: "STRING", description: "Optional. The new key for the entry."}, new_value: {type: "STRING", description: "Optional. The new value for the entry."} }, required: ["key"] } }, { name:"link_data", description:"Creates a link between two entries (data or files) in the graph store.", parameters:{type:"OBJECT",properties:{source_key:{type:"STRING", description: "Source entry key."},target_key:{type:"STRING", description: "Target entry key."}},required:["source_key","target_key"]}}, { name:"delete_data", description:"Deletes a data entry by key from the graph store. This cannot delete files; use \`delete_file\` for that.", parameters:{type:"OBJECT",properties:{key:{type:"STRING", description: "The key of the data entry to delete."}},required:["key"]}}, { name:"search_data", description:"Searches and highlights entries (both data and files) in the graph store based on a query.", parameters:{type:"OBJECT",properties:{query:{type:"STRING", description: "The search term."}},required:["query"]}}, { name:"clear_store", description:"Erases the entire graph data store, including all data entries and files.", parameters:{type:"OBJECT",properties:{}}}, { name:"list_data", description:"Lists all data entries (not files) from the graph store.", parameters:{type:"OBJECT",properties:{}}}, { name: "web_search", description: "Performs a web search using DuckDuckGo and returns the top 10 results.", parameters: { type: "OBJECT", properties: { query: { type: "STRING", description: "The search query." } }, required: ["query"] } }, { name: "fetch_page_content", description: "Fetches the clean, visible text content from a given URL. Useful for reading articles or web pages found via web_search.", parameters: { type: "OBJECT", properties: { url: { type: "STRING", description: "The full URL of the web page to read." } }, required: ["url"] } }, { name: "display_chart", description: "Renders data as a chart. Supports types like 'bar', 'line', 'pie', 'doughnut', 'radar', and 'polarArea'.", parameters: { type: "OBJECT", properties: { type: { type: "STRING", description: "The type of chart to display. Common options: 'bar', 'line', 'pie', 'doughnut', 'radar'." }, data: { type: "OBJECT", description: "The data for the chart, following Chart.js structure.", properties: { labels: { type: "ARRAY", description: "An array of strings for the x-axis or segment labels.", items: { type: "STRING" } }, datasets: { type: "ARRAY", description: "An array of dataset objects to plot.", items: { type: "OBJECT", properties: { label: { type: "STRING", description: "The label for this dataset (appears in legend and tooltips)." }, data: { type: "ARRAY", description: "The numerical data points for this dataset.", items: { type: "NUMBER" } }, backgroundColor: { type: "ARRAY", description: "Optional. Background color(s) for the data points. Can be a single color string or an array of strings.", items: { type: "STRING" } }, borderColor: { type: "ARRAY", description: "Optional. Border color(s) for the data points. Can be a single color string or an array of strings.", items: { type: "STRING" } } }, required: ["label", "data"] } } }, required: ["labels", "datasets"] }, options: { type: "OBJECT", description: "Optional. Configuration options for the chart.", properties: { title: { type: "STRING", description: "The main title to display above the chart." } } } }, required: ["type", "data"] } }, { name: "execute_javascript", description: "Executes JavaScript code in a secure sandbox with file system access. Use the \`M8.files\` object to read/write file nodes in the store.", parameters: { type: "OBJECT", properties: { code: { type: "STRING", description: "The JavaScript code to execute. Must use \`return\` for output. Access files via \`M8.files.read/write\`." }, timeout: { type: "NUMBER", description: "Optional. The maximum execution time in milliseconds. Defaults to 5000." } }, required: ["code"] } }, { name: "list_files", description: "Lists the names of all files currently stored in the unified graph store.", parameters: { type: "OBJECT", properties: {} } }, { name: "read_file", description: "Reads the content of a specified file from the unified store.", parameters: { type: "OBJECT", properties: { filename: { type: "STRING", description: "The name of the file to read." } }, required: ["filename"] } }, { name: "delete_file", description: "Deletes a specified file from the unified store. This cannot delete regular data entries.", parameters: { type: "OBJECT", properties: { filename: { type: "STRING", description: "The name of the file to delete." } }, required: ["filename"] } }, { name: "create_download_link", description: "Presents a file to the user for download. If 'content' is provided, it creates the file on the fly. Otherwise, it uses an existing file specified by 'filename' from the unified store.", parameters: { type: "OBJECT", properties: { filename: { type: "STRING", description: "The name of the file for the user to download." }, content: { type: "STRING", description: "Optional. The text content of the file. If omitted, the tool will try to find a file with the given 'filename' in the store." } }, required: ["filename"] } } ]}];
                const isGraphTool = (name) => ["store_data", "update_data", "link_data", "delete_data", "search_data", "clear_store", "list_data"].includes(name);
                
                const proxy = 'https://api.allorigins.win/raw?url=';

                async function doSearch(q) {
                  const ddgHTML = 'https://html.duckduckgo.com/html/?q=';
                  const fetchUrl = proxy + encodeURIComponent(ddgHTML + encodeURIComponent(q));
                  const res = await fetch(fetchUrl);
                  if (!res.ok) throw new Error('Fetch failed: ' + res.status);
                  const text = await res.text();
                  const parser = new DOMParser();
                  const doc = parser.parseFromString(text, 'text/html');

                  const links = [...doc.querySelectorAll('a.result__a')].slice(0, 10);
                  const snippets = [...doc.querySelectorAll('a.result__snippet')].slice(0, 10);

                  return links.map((a, i) => ({
                    title: a.textContent.trim(),
                    href: a.href,
                    snippet: snippets[i]?.textContent.trim() || '',
                  }));
                }

                async function fetchPageText(url) {
                  try {
                    const res = await fetch(proxy + encodeURIComponent(url));
                    if (!res.ok) throw new Error('Failed to load page: ' + res.status);
                    const htmlText = await res.text();
                    const parser = new DOMParser();
                    const doc = parser.parseFromString(htmlText, 'text/html');
                    [...doc.querySelectorAll('script, style, noscript, iframe, meta, link, [hidden]')].forEach(el => el.remove());
                    const text = doc.body.innerText.trim();
                    return text || 'No visible text content found.';
                  } catch (err) {
                    return 'Error fetching page content: ' + err.message;
                  }
                }

                const handleWeatherForecast = async (args) => { const { location, date } = args; if (!location) return { success: false, message: "A location must be provided." }; try { const r = await fetch(\`https://wttr.in/\${encodeURIComponent(location)}?format=j1\`); if (!r.ok) { if (r.status === 404) return { success: false, message: \`Could not find weather for: "\${location}".\` }; throw new Error(\`Network error. Status: \${r.status}\`); } const w = await r.json(); const lN = \`\${w.nearest_area[0].areaName[0].value}, \${w.nearest_area[0].country[0].value}\`; if (date) { if (!/^\\d{4}-\\d{2}-\\d{2}$/.test(date)) return { success: false, message: "Invalid date format. Use YYYY-MM-DD." }; const dD = w.weather.find(d => d.date === date); if (!dD) return { success: false, message: \`Forecast for \${date} is not available.\` }; const hF = dD.hourly.map(h => { const h24 = parseInt(h.time) / 100; const p = h24 >= 12 ? 'PM' : 'AM'; let h12 = h24 % 12; if (h12 === 0) h12 = 12; return { time: \`\${h12} \${p}\`, temp_C: \`\${h.tempC}°C\`, feels_like_C: \`\${h.FeelsLikeC}°C\`, description: h.weatherDesc[0].value, chance_of_rain: \`\${h.chanceofrain}%\` }; }); return { success: true, data: { location: lN, date: date, hourly_forecast: hF } }; } else { const c = w.current_condition[0]; const t = w.weather[0]; const tm = w.weather[1]; const s = { location: lN, current_conditions: { description: c.weatherDesc[0].value, temp_C: \`\${c.temp_C}°C\`, feels_like_C: \`\${c.FeelsLikeC}°C\`, humidity: \`\${c.humidity}%\`, wind: \`\${c.windspeedKmph} km/h \${c.winddir16Point}\` }, today_forecast: { date: t.date, max_temp_C: \`\${t.maxtempC}°C\`, min_temp_C: \`\${t.mintempC}°C\`, summary: t.hourly[4].weatherDesc[0].value, sunrise: t.astronomy[0].sunrise, sunset: t.astronomy[0].sunset }, tomorrow_forecast: { date: tm.date, max_temp_C: \`\${tm.maxtempC}°C\`, min_temp_C: \`\${tm.mintempC}°C\`, summary: tm.hourly[4].weatherDesc[0].value } }; return { success: true, data: s }; } } catch (e) { console.error("Weather fetch failed:", e); return { success: false, message: \`Error fetching weather: \${e.message}\` }; } };
                
                const handleToolCall = async (functionName, args) => {
                    let result;
                    switch (functionName) {
                        case "display_chart":
                            const { type, data, options } = args;
                            if (!type || !data || !data.labels || !data.datasets) {
                                result = { success: false, message: "Invalid chart data provided. 'type', 'data.labels', and 'data.datasets' are required." };
                            } else {
                                const chartConfig = { type, data, options: options || {} };
                                addOrUpdateMessage('system', '', null, false, false, '[Chart displayed]', false, false, false, false, true, chartConfig);
                                result = { success: true, message: "Chart has been displayed to the user." };
                            }
                            break;
                        case "web_search":
                            try {
                                const searchResults = await doSearch(args.query);
                                result = { success: true, message: \`Found \${searchResults.length} results.\`, data: { results: searchResults } };
                            } catch (e) {
                                console.error("Web search failed:", e);
                                result = { success: false, message: \`Error performing web search: \${e.message}\` };
                            }
                            break;
                        case "fetch_page_content":
                            const pageText = await fetchPageText(args.url);
                            if (pageText.startsWith('Error fetching page content:')) {
                                result = { success: false, message: pageText };
                            } else {
                                const truncatedText = pageText.length > 15000 ? pageText.substring(0, 15000) + '... [truncated]' : pageText;
                                result = { success: true, message: "Page content fetched successfully.", data: { content: truncatedText } };
                            }
                            break;
                        case "execute_javascript":
                            const filesForWorker = visGraphManager.nodes.get({ filter: n => n.type === 'file' }).reduce((acc, node) => { acc[node.id] = node.body; return acc; }, {});
                            const workerCode = \`
                                self.onmessage = function(e) {
                                    const { code, files } = e.data;
                                    const M8 = {
                                        files: {
                                            _data: files,
                                            read(filename) { if (typeof filename !== 'string') return undefined; return this._data[filename]; },
                                            write(filename, content) { if (typeof filename !== 'string') return; this._data[filename] = String(content); },
                                            list() { return Object.keys(this._data); }
                                        }
                                    };
                                    try {
                                        const func = new Function('M8', \`return (function() { \${code} })();\`);
                                        const result = func(M8);
                                        self.postMessage({ success: true, data: result, updatedFiles: M8.files._data });
                                    } catch (err) { self.postMessage({ success: false, message: err.message }); }
                                };
                            \`;
                            const blob = new Blob([workerCode], { type: 'application/javascript' });
                            const worker = new Worker(URL.createObjectURL(blob));
                            const executionPromise = new Promise((resolve) => {
                                const timeout = args.timeout || 5000;
                                const timer = setTimeout(() => { worker.terminate(); resolve({ success: false, message: \`Execution timed out after \${timeout}ms.\` }); }, timeout);
                                worker.onmessage = (e) => { clearTimeout(timer); worker.terminate(); resolve(e.data); };
                                worker.onerror = (e) => { clearTimeout(timer); worker.terminate(); resolve({ success: false, message: \`An error occurred in the sandboxed code: \${e.message}\`}); };
                                worker.postMessage({ code: args.code, files: filesForWorker });
                            });
                            try {
                                const execRes = await executionPromise;
                                if (execRes.success) {
                                    const originalFilenames = new Set(Object.keys(filesForWorker));
                                    const updatedFilenames = new Set(Object.keys(execRes.updatedFiles));
                                    for (const filename of updatedFilenames) { visGraphManager.nodes.update({ id: filename, label: filename, body: execRes.updatedFiles[filename], type: 'file' }); }
                                    for (const filename of originalFilenames) { if (!updatedFilenames.has(filename)) { visGraphManager.nodes.remove(filename); } }
                                    saveGraphToLocalStorage();
                                    const dataToReturn = typeof execRes.data === 'object' ? JSON.stringify(execRes.data, null, 2) : execRes.data;
                                    result = { success: true, data: { result: dataToReturn }, message: "Code executed successfully." };
                                } else { result = { success: false, message: \`Code execution failed: \${execRes.message}\` }; }
                            } catch (e) { result = { success: false, message: e.message }; }
                            break;
                        
                        case "list_files":
                            const fileNodes = visGraphManager.nodes.get({ filter: n => n.type === 'file' });
                            const filenames = fileNodes.map(n => n.id);
                            if (filenames.length === 0) { result = { success: true, message: "No files are currently stored." }; }
                            else { result = { success: true, message: \`Found \${filenames.length} file(s).\`, data: filenames }; }
                            break;
                        case "read_file":
                            const nodeToRead = visGraphManager.nodes.get(args.filename);
                            if (nodeToRead && nodeToRead.type === 'file') { result = { success: true, message: \`Content of "\${args.filename}" retrieved.\`, data: { content: nodeToRead.body } }; }
                            else if (nodeToRead) { result = { success: false, message: \`Entry "\${args.filename}" found, but it is not a file.\` }; }
                            else { result = { success: false, message: \`File "\${args.filename}" not found.\` }; }
                            break;
                        case "delete_file":
                            const nodeToDelete = visGraphManager.nodes.get(args.filename);
                            if (nodeToDelete && nodeToDelete.type === 'file') { result = visGraphManager.deleteNode(args.filename); }
                            else if (nodeToDelete) { result = { success: false, message: \`Cannot delete "\${args.filename}" because it is not a file. Use delete_data instead.\` }; }
                            else { result = { success: false, message: \`File "\${args.filename}" not found.\` }; }
                            break;
                         case "create_download_link":
                            const { filename, content: newContent } = args;
                            let fileContent;
                            if (newContent !== undefined) { fileContent = newContent; }
                            else { const nodeForLink = visGraphManager.nodes.get(filename); if (nodeForLink && nodeForLink.type === 'file') { fileContent = nodeForLink.body; } }
                            if (fileContent === undefined) { result = { success: false, message: \`File "\${filename}" not found in store and no content was provided.\` }; } 
                            else { const blob = new Blob([fileContent], { type: 'text/plain;charset=utf-8' }); const url = URL.createObjectURL(blob); const downloadLinkHtml = \`<p class="font-semibold mb-2 text-slate-700 dark:text-slate-200">Download Ready:</p><a href="\${url}" download="\${filename}" class="inline-flex items-center gap-2 px-3 py-1.5 bg-zinc-800 text-white rounded-md hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-500 dark:bg-zinc-700 dark:hover:bg-zinc-600 text-sm font-medium"><span class="material-symbols-outlined !text-base">download</span> Download \${filename}</a>\`; addOrUpdateMessage('system', downloadLinkHtml, null, false, false, \`[Download link for \${filename}]\`, true, false, false, true); result = { success: true, message: \`Successfully created a download link for "\${filename}".\` }; }
                            break;
                        case "get_weather_forecast": result = await handleWeatherForecast(args); break;
                        case "store_data": result = visGraphManager.addNode(args.key, args.value, { type: 'data' }); break;
                        case "update_data": result = visGraphManager.editNode(args.key, args.new_key, args.new_value); break;
                        case "link_data": result = visGraphManager.addLink(args.source_key, args.target_key); break;
                        case "delete_data": 
                            const nodeTypeCheck = visGraphManager.nodes.get(args.key);
                            if (nodeTypeCheck && nodeTypeCheck.type === 'file') { result = { success: false, message: \`Cannot delete "\${args.key}" because it is a file. Use delete_file instead.\`}; }
                            else { result = visGraphManager.deleteNode(args.key); }
                            break;
                        case "search_data": result = visGraphManager.highlightSearch(args.query); break;
                        case "clear_store": result = visGraphManager.clearGraphData(); break;
                        case "list_data": const dataNodes = visGraphManager.nodes.get({ filter: n => n.type !== 'file' }).map(n => ({ id: n.id, body: n.body })); result = { success: true, message: \`Found \${dataNodes.length} data entries.\`, data: dataNodes }; if (result.success) visGraphManager.zoomToFitAllNodes(); break;
                        default: result = { success: false, message: \`Unknown function: \${functionName}\` };
                    }
                    if (result.success && isGraphTool(functionName)) {
                        if (activeTempGraphSlotId.value) { const i = activeTab.value.messages.findIndex(m => m.id === activeTempGraphSlotId.value); if (i > -1) activeTab.value.messages.splice(i, 1); activeTempGraphSlotId.value = null; }
                        const newId = 'graph-slot-' + Date.now();
                        addOrUpdateMessage('system', '', newId, false, false, '', false, true, false, false);
                        activeTempGraphSlotId.value = newId;
                        const dur = functionName === 'list_data' ? 9000 : 7000;
                        setTimeout(() => { if (activeTempGraphSlotId.value === newId) { const i = activeTab.value.messages.findIndex(m => m.id === newId); if (i > -1) activeTab.value.messages.splice(i, 1); activeTempGraphSlotId.value = null; if (functionName === 'list_data') visGraphManager.resetZoom(); } }, dur);
                    }
                    return result;
                };
                const triggerFileInput = () => { fileInputRef.value.click(); };
                const handleFileUpload = (event) => {
                    const files = event.target.files;
                    if (!files.length) return;
                    let uploadedFiles = [];
                    for (const file of files) {
                        const reader = new FileReader();
                        reader.onload = (e) => {
                            const content = e.target.result;
                            const filename = file.name;
                            visGraphManager.nodes.update({ id: filename, label: filename, body: content, type: 'file', title: \`File: \${filename}\\nSize: \${content.length} bytes\` });
                            uploadedFiles.push(filename);
                            if (uploadedFiles.length === files.length) {
                                 saveGraphToLocalStorage();
                                 const fileNames = uploadedFiles.map(f => \`"\${f}"\`).join(', ');
                                 addOrUpdateMessage('system', \`Successfully uploaded/updated \${uploadedFiles.length} file(s): \${fileNames}\`, null, false, false, \`Uploaded \${fileNames}\`, true, false, false, true);
                            }
                        };
                        reader.onerror = (e) => { addOrUpdateMessage('system', \`Error reading file "\${file.name}".\`, null, false, true, \`Error reading \${file.name}\`, true); };
                        reader.readAsText(file);
                    }
                    event.target.value = '';
                };

                const autoGrowTextarea=(ev)=>{const el=ev.target;el.style.height='auto';el.style.height=\`\${Math.min(el.scrollHeight,128)}px\`;};
                const scrollToBottom=(force=false)=>{nextTick(()=>{if(chatContainerRef.value){const{scrollTop,scrollHeight,clientHeight}=chatContainerRef.value;if(force||scrollHeight-scrollTop-clientHeight<150)chatContainerRef.value.scrollTop=scrollHeight;}});};
                const addOrUpdateMessage=(role,text,id=null,stream=false,err=false,origMd=null,sysTool=false,graphSlot=false,isCodeDisplay=false,isRawHtml=false,isChartDisplaySlot=false,chartConfig=null)=>{ if(!activeTab.value)return;const messages=activeTab.value.messages;const exIdx=id?messages.findIndex(m=>m.id===id):-1;const msgData={id:id||Date.now().toString(36)+Math.random().toString(36).substring(2),role:role,parts:[{text:text}],originalMarkdown:origMd!==null?origMd:text,isStreaming:stream,isError:err,isSystemToolResponse:sysTool,isGraphDisplaySlot:graphSlot,isCodeDisplay:isCodeDisplay,isRawHtml:isRawHtml,isChartDisplaySlot:isChartDisplaySlot,chartConfig:chartConfig,timestamp:Date.now()};if(exIdx>-1)messages.splice(exIdx,1,msgData);else messages.push(msgData);scrollToBottom(role!=='model'||sysTool||graphSlot||isChartDisplaySlot);return msgData.id;};
                const handleSendMessage=async()=>{const txt=newMessageText.value.trim();if(!txt&&!editingMessage.value)return;isLoading.value=true;const ta=document.getElementById('-input');if(editingMessage.value){const idx=activeTab.value.messages.findIndex(m=>m.id===editingMessage.value.id);if(idx>-1){activeTab.value.messages[idx].parts[0].text=txt;activeTab.value.messages[idx].originalMarkdown=txt;}editingMessage.value=null;isLoading.value=false;}else{if(activeTab.value){activeTab.value.activeTools=[];}addOrUpdateMessage('user',txt);const hist=activeTab.value.messages.filter(m=>!m.isError&&!m.isSystemToolResponse&&!m.isGraphDisplaySlot&&!m.isChartDisplaySlot).map(m=>{const parts=m.role==='tool'?m.parts:m.parts.map(p=>({text:p.text}));return{role:m.role,parts:parts};});await fetchBotResponse(hist);}newMessageText.value='';if(ta)ta.style.height='auto';if(!editingMessage.value&&ta)ta.focus();};
                const fetchBotResponse=async(chatHist)=>{ if (!activeTab.value) { isLoading.value = false; return; } const botMsgId=addOrUpdateMessage('model','',null,true); isLoading.value=true; const currentDate = new Date().toISOString().slice(0, 10); const dynamicSystemPrompt = \`\${SYSTEM_PROMPT}\\n\\n# Current Context\\n- The current date is: \${currentDate}.\`; const payload = { contents: chatHist, tools: geminiTools, systemInstruction: { parts: [{ text: dynamicSystemPrompt }] } }; try{const resp=await fetch(apiUrl.value,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)}); const msgIdx=activeTab.value.messages.findIndex(m=>m.id===botMsgId); if(msgIdx===-1){isLoading.value=false;return;}if(!resp.ok){const errData=await resp.json().catch(()=>({error:{message:"API error"}}));const errMsg=errData.error?.message||\`Err: \${resp.status}\`;activeTab.value.messages[msgIdx].parts[0].text=\`<p>API Error:</p><p>\${errMsg.replace(/\\n/g,'<br>')}</p>\`;activeTab.value.messages[msgIdx].isError=true;activeTab.value.messages[msgIdx].isStreaming=false;if(activeTab.value){activeTab.value.activeTools = [];} isLoading.value=false; return;}const data=await resp.json();if(!data.candidates||!data.candidates[0]||!data.candidates[0].content||!data.candidates[0].content.parts){activeTab.value.messages[msgIdx].parts[0].text=\`<p>Error:</p><p>Invalid AI response.</p>\`;activeTab.value.messages[msgIdx].isError=true;activeTab.value.messages[msgIdx].isStreaming=false;if(activeTab.value){activeTab.value.activeTools = [];} isLoading.value=false; return;}const parts=data.candidates[0].content.parts;let textResp="",funcCalls=[];for(const p of parts){if(p.text)textResp+=p.text;else if(p.functionCall)funcCalls.push(p.functionCall);}if(funcCalls.length>0){ if (activeTab.value) { const newTools = funcCalls.map((fc, i) => ({ name: fc.name, status: i === 0 ? 'running' : 'pending' })); activeTab.value.activeTools.push(...newTools); } const toolResponses=[]; const startIndex = activeTab.value.activeTools.length - funcCalls.length; for (const [index, funcCall] of funcCalls.entries()) { const toolIndexInTab = startIndex + index; if (funcCall.name === 'execute_javascript' && funcCall.args.code) { const codeContent = '\\\`\\\`\\\`javascript\\n' + funcCall.args.code + '\\n\\\`\\\`\\\`'; const displayText = \`▶️ **Executing Code**\\n\${codeContent}\`; addOrUpdateMessage('model', displayText, null, false, false, displayText, false, false, true, false); } const funcRes = await handleToolCall(funcCall.name, funcCall.args); if (funcCall.name !== 'create_download_link' && funcCall.name !== 'display_chart') { addOrUpdateMessage('system',\`<b>[\${funcCall.name}]:</b> \${funcRes.message?.replace(/\\n/g,'<br>') || 'Completed.'}\`,null,false,!funcRes.success,\`[Res: \${funcRes.message || 'Completed.'}]\`,true, false, false, funcRes.isRawHtml); } let modelContent=funcRes;if(funcRes.success&&funcRes.data)modelContent=funcRes.data;toolResponses.push({functionResponse:{name:funcCall.name,response:{name:funcCall.name,content:modelContent}}});if (activeTab.value && activeTab.value.activeTools[toolIndexInTab]) { activeTab.value.activeTools[toolIndexInTab].status = 'success'; if (activeTab.value.activeTools[toolIndexInTab + 1]) { activeTab.value.activeTools[toolIndexInTab + 1].status = 'running'; } } } const originalMessageIndex = activeTab.value.messages.findIndex(m => m.id === botMsgId); if (originalMessageIndex > -1) activeTab.value.messages.splice(originalMessageIndex, 1); const newHist=[...chatHist,{role:'model',parts:funcCalls.map(fc=>({functionCall:fc}))},{role:'tool',parts:toolResponses}]; await fetchBotResponse(newHist); }else if(textResp.trim()!==""){activeTab.value.messages[msgIdx].originalMarkdown=textResp;let curTxt="",charIdx=0;const streamInt=setInterval(()=>{if(charIdx<textResp.length){curTxt+=textResp[charIdx++];activeTab.value.messages[msgIdx].parts[0].text=curTxt;scrollToBottom();}else{clearInterval(streamInt);activeTab.value.messages[msgIdx].parts[0].text=textResp;activeTab.value.messages[msgIdx].isStreaming=false;scrollToBottom(true);isLoading.value=false; }},15);}else{ if(data.candidates[0].finishReason==="SAFETY"||data.promptFeedback?.blockReason){const reason=data.promptFeedback?.blockReason||"Safety";activeTab.value.messages[msgIdx].parts[0].text=\`<p>Blocked:</p><p>Reason: \${reason}.</p>\`;activeTab.value.messages[msgIdx].isError=true;}else{activeTab.value.messages[msgIdx].parts[0].text=\`\`;} activeTab.value.messages[msgIdx].isStreaming=false;isLoading.value=false;}}catch(err){console.error('Fetch Error:',err);const mIdx=activeTab.value.messages.findIndex(m=>m.id===botMsgId);if(mIdx>-1){activeTab.value.messages[mIdx].parts[0].text=\`<p>Error:</p><p>\${err.message}</p>\`;activeTab.value.messages[mIdx].isError=true;activeTab.value.messages[mIdx].isStreaming=false;} if (activeTab.value) { activeTab.value.activeTools = []; } isLoading.value=false;}finally{const fIdx=activeTab.value.messages.findIndex(m=>m.id===botMsgId);if(fIdx>-1&&activeTab.value.messages[fIdx].isStreaming){activeTab.value.messages[fIdx].isStreaming=false;}if(!activeTab.value.messages.some(m=>m.isStreaming&&m.role==='model'))isLoading.value=false;scrollToBottom(true);}};
                const startEdit=(msg)=>{editingMessage.value=msg;newMessageText.value=msg.originalMarkdown;const ta=document.getElementById('-input');if(ta){ta.focus();nextTick(()=>autoGrowTextarea({target:ta}));}};
                const performCopy=(msg)=>{const txt=msg.originalMarkdown||msg.parts[0].text;navigator.clipboard.writeText(txt.trim()).then(()=>{const el=document.querySelector(\`[data-message-id="\${msg.id}"]\`);if(el){const btn=el.querySelector('.copy-button');if(btn){const origHTML=btn.innerHTML;btn.innerHTML='<span class="material-symbols-outlined !text-sm !leading-none text-green-500">check</span>';setTimeout(()=>{btn.innerHTML=origHTML;},1500);}}}).catch(err=>console.error('Copy fail:',err));};
                
                onMounted(() => {
                    const savedKey = localStorage.getItem(LS_API_KEY_KEY);
                    const savedModel = localStorage.getItem(LS_MODEL_NAME_KEY);
                    if (savedKey) apiKey.value = savedKey;
                    if (savedModel) modelName.value = savedModel;
                    const isApiKeyMissing = !apiKey.value;

                    visGraphManager.setSyncCallback(saveGraphToLocalStorage);
                    const savedGraphData = localStorage.getItem(LS_GRAPH_DATA_KEY);
                    if (savedGraphData) { try { visGraphManager.loadTabData(JSON.parse(savedGraphData)); } catch(e) { console.error("Could not load shared graph data:", e); } }

                    const savedSessions = localStorage.getItem(LS_SESSIONS_KEY);
                    if (savedSessions) {
                        try {
                            const parsedTabs = JSON.parse(savedSessions);
                            if (Array.isArray(parsedTabs) && parsedTabs.length > 0) {
                                tabs.value = parsedTabs.map(t => ({...t, activeTools: [], messages: t.messages.map(m => ({...m, isRawHtml: m.isRawHtml || false, isChartDisplaySlot: m.isChartDisplaySlot || false, chartConfig: m.chartConfig || null})) }));
                                activeTabId.value = parsedTabs[0].id;
                            } else { addNewTab(); }
                        } catch(e) { addNewTab(); }
                    } else { addNewTab(); }
                    
                    if (isApiKeyMissing) {
                        openSettingsModal();
                        const welcomeMessage = "Welcome! Please provide your Gemini API key in the Settings panel (click 'M8' in the header). You can get a free key from Google AI Studio.";
                        if(activeTab.value && activeTab.value.messages.length === 1 && activeTab.value.messages[0].originalMarkdown === '👋') {
                            activeTab.value.messages[0].parts[0].text = welcomeMessage;
                            activeTab.value.messages[0].originalMarkdown = welcomeMessage;
                        }
                    }
                    const darkModeObserver = new MutationObserver((mutations) => { 
                        for (const mutation of mutations) {
                            if (mutation.attributeName === 'class') {
                                visGraphManager.updateTheme();
                                // Manually trigger a class update on the main container
                                const container = document.getElementById('m8-app-container');
                                if (container) {
                                    if(document.documentElement.classList.contains('dark')) {
                                        container.classList.add('dark');
                                    } else {
                                        container.classList.remove('dark');
                                    }
                                }
                            }
                        }
                    });
                    const container = document.getElementById('m8-app-container');
                    if(container) darkModeObserver.observe(container, { attributes: true });

                    onUnmounted(() => { darkModeObserver.disconnect(); visGraphManager.destroyVisualization(); });
                });

                return { tabs, activeTabId, activeTab, addNewTab, selectTab, closeTab, newMessageText, isLoading, handleSendMessage, startEdit, performCopy, chatContainerRef, editingMessage, autoGrowTextarea, showSettingsModal, settingsModel, settingsApiKey, openSettingsModal, closeSettingsModal, saveSettings, fileInputRef, triggerFileInput, handleFileUpload, getToolIcon };
              }
            }).mount('#m8-app-container #app'); // MODIFICATION 3: Mount to the injected container
            `;

            appScript.innerHTML = originalScriptContent;
            document.body.appendChild(appScript);
            
            // --- G. ADD WINDOW FUNCTIONALITY (DRAG, CLOSE, MINIMIZE) ---
            document.getElementById('m8-close-btn').addEventListener('click', () => {
                appContainer.remove();
                style.remove();
                appScript.remove();
                document.getElementById('m8-loader-script').remove(); // remove self to allow re-injection
            });

             document.getElementById('m8-hide-btn').addEventListener('click', () => {
                appContainer.style.display = 'none';
            });

            const header = document.getElementById('m8-header');
            let isDragging = false;
            let offsetX, offsetY;
            header.addEventListener('mousedown', (e) => {
                // Prevent dragging when clicking on buttons
                if (e.target.tagName === 'BUTTON') return;
                isDragging = true;
                const rect = appContainer.getBoundingClientRect();
                offsetX = e.clientX - rect.left;
                offsetY = e.clientY - rect.top;
                document.body.style.userSelect = 'none';
            });
            document.addEventListener('mousemove', (e) => {
                if (!isDragging) return;
                e.preventDefault();
                appContainer.style.left = `${e.clientX - offsetX}px`;
                appContainer.style.top = `${e.clientY - offsetY}px`;
            });
            document.addEventListener('mouseup', () => {
                isDragging = false;
                document.body.style.userSelect = '';
            });

            // --- H. FINAL CLEANUP ---
            loadingIndicator.remove();
            console.log("M8: Initialization complete.");

        } catch (error) {
            console.error("M8: Failed to initialize.", error);
            loadingIndicator.innerText = 'Error loading M8!';
            loadingIndicator.style.backgroundColor = '#dc2626'; // Red
            // Keep the indicator on screen to show the error
        }
    }

    // --- KICK OFF THE PROCESS ---
    if (document.readyState === 'complete' || document.readyState === 'interactive') {
        initializeM8();
    } else {
        window.addEventListener('DOMContentLoaded', initializeM8);
    }

})();
