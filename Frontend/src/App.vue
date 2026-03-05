<template>
  <div
    class="flex flex-col justify-center items-center w-full h-screen font-sans overflow-hidden"
  >
    <TransitionGroup name="chat-transition">
      <div
        v-if="chatMessages.length === 0"
        key="initial-state"
        class="flex flex-col items-center w-full"
      >
        <div class="text-4xl font-bold mb-4 text-center text-base-content">
          <img src="/logo.svg" alt="Flui3d Chat" class="h-60 w-72 object-contain">
        </div>
        <div class="flex flex-col w-[90%] max-w-3xl bg-neutral rounded-2xl p-2">
          <textarea
            v-model="userMessage"
            placeholder="Describe your microfluidic chip design needs..."
            @keydown.enter.exact.prevent="sendMessage"
            @input="adjustHeight"
            class="p-2 border-none bg-neutral text-neutral-content text-base resize-none outline-none min-h-[110px] max-h-[260px]"
            ref="textarea"
          ></textarea>

          <div class="flex justify-between items-center mt-2">
            <div
              v-if="!isLoading"
              class="self-end"
              :class="[isLoading ? '' : 'tooltip tooltip-secondary tooltip-right lg:tooltip-bottom']"
              :data-tip="[reasoningEnabled ? 'Disable reasoning' : 'Enable reasoning']"
            >
              <button
                @click="toggleReasoning"
                class="btn btn-circle btn-ghost btn-sm"
                :class="[
                  reasoningEnabled ? 'text-pink-300' : 'text-base-content',
                  isLoading ? 'pointer-events-none cursor-default' : '',
                ]"
              >
                <font-awesome-icon icon="fa-regular fa-lightbulb" size="lg" />
              </button>
            </div>

            <button
              @click="sendMessage"
              :disabled="isLoading || !userMessage"
              class="w-[40px] h-[40px] rounded-full bg-cyan-400 text-neutral text-2xl font-bold flex justify-center items-center transition-colors duration-500 ease-in-out disabled:bg-neutral-content"
            >
              <span v-if="!isLoading">
                <font-awesome-icon icon="fa-solid fa-arrow-right" size="xs" />
              </span>
              <span
                v-else
                class="w-4 h-4 border-2 border-neutral border-t-transparent rounded-full animate-spin"
              ></span>
            </button>
          </div>
        </div>
      </div>

      <div v-else key="chat-state" class="flex flex-col items-center w-full h-full">
        <!-- Menu Bar -->
        <div class="bg-white flex w-full h-fit">
          <div class="flex items-center w-full px-[5%] my-4">
            <!-- Buttons Left -->
            <div class="flex w-full h-full justify-start">
              <div v-if="selectedMessagePair" class="flex">
                <!-- Conversation History Button -->
                <div v-if="selectedView === 0" class="flex">
                  <div
                    class="tooltip tooltip-neutral tooltip-right lg:tooltip-bottom"
                    data-tip="Conversation history"
                  >
                    <button
                      @click="switchView"
                      class="btn btn-ghost text-base-content hover:text-pink-400"
                    >
                      <font-awesome-icon icon="fa-solid fa-comments" size="lg" />
                    </button>
                  </div>
                </div>

                <!-- Latest Chip Design Button -->
                <div
                  v-else
                  class="tooltip tooltip-neutral tooltip-right lg:tooltip-bottom"
                  data-tip="Latest chip design"
                >
                  <button
                    @click="switchView"
                    class="btn btn-ghost text-base-content hover:text-pink-400"
                  >
                    <font-awesome-icon icon="fa-solid fa-images" size="lg" />
                  </button>
                </div>
              </div>
            </div>

            <!-- Title -->

            <div class="text-2xl font-bold text-base-content">
              <img src="/logo-h.svg" alt="Flui3d Chat" class="h-16 w-72 object-contain">
            </div>

            <!-- Buttons Right -->

            <!-- Settings Button -->
            <div class="flex w-full h-full justify-end">
              <div class="dropdown dropdown-bottom dropdown-end">
                <div class="tooltip tooltip-neutral tooltip-bottom" data-tip="Settings">
                  <label tabindex="0" class="btn btn-ghost text-base-content hover:text-cyan-400"
                    ><font-awesome-icon icon="fa-solid fa-gear" size="lg"
                  /></label>
                </div>
                <div
                  tabindex="0"
                  class="dropdown-content z-[1] card card-compact w-64 p-2 shadow bg-base-200 text-base-content"
                >
                  <div class="card-body max-h-80 overflow-y-auto">
                    <h3 class="card-title">Settings</h3>
                    <div v-for="(setting, key) in settings" :key="key">
                      <label class="label">
                        <span class="label-text">{{ setting.label }}</span>
                      </label>
                      <input
                        type="number"
                        class="input input-xs w-full max-w-xs"
                        v-model.number="setting.value"
                      />
                    </div>
                  </div>
                  <div class="px-4 pb-2 pt-4">
                    <button
                      class="btn btn-sm w-full bg-[#B685D2] hover:bg-[#9E6EBA] text-white border-none"
                      @click="resynthesizeChip"
                    >
                      Save & Resynthesize
                    </button>
                  </div>
                </div>
              </div>

              <!-- Reset Conversation Button -->
              <div
                class="tooltip tooltip-neutral tooltip-left lg:tooltip-bottom"
                data-tip="Reset conversation"
              >
                <button
                  onclick="reset_conversation_modal.showModal()"
                  class="btn btn-ghost text-base-content hover:text-error"
                >
                  <font-awesome-icon icon="fa-solid fa-rotate-left" size="lg" />
                </button>
              </div>
            </div>
          </div>
        </div>

        <!-- Main View Container -->
        <div class="flex flex-col grow w-full min-h-[200px] items-center">
          <!-- Chip View Container -->
          <div
            v-if="selectedView === 0 && selectedMessagePair"
            class="flex w-full h-full pl-[5%] pr-[5%] pb-4 items-start transition-transform duration-300 ease-in-out transform-gpu"
            :class="{
              'translate-x-0 md:translate-x-0':
                !selectedMessagePair.assistant ||
                selectedMessagePair.assistant.activePanel === 'code',
              '-translate-x-full md:translate-x-0':
                selectedMessagePair.assistant &&
                selectedMessagePair.assistant.activePanel === 'svg',
            }"
          >
            <!-- LLM input and output -->
            <div
              class="flex w-full md:min-w-[400px] md:w-min shrink-0 h-full relative transition-all duration-300"
            >
              <div
                class="flex flex-col w-full h-full bg-neutral overflow-hidden rounded-xl relative"
              >
                <!-- User Message -->
                <div
                  class="h-auto max-h-[150px] bg-primary text-white text-sm drop-shadow-xl"
                >
                  <div class="h-full overflow-y-auto py-2 px-4">
                    <span>
                      {{ selectedMessagePair.user.content }}
                    </span>
                  </div>
                </div>

                <!-- Reasoning Animation -->
                <div
                  v-if="
                    selectedMessagePair.assistant &&
                    selectedMessagePair.assistant.isThinking &&
                    !selectedMessagePair.assistant.isOptimizing
                  "
                  class="flex-1 text-neutral-content flex flex-col items-center justify-center text-center"
                >
                  <font-awesome-icon
                    icon="fa-regular fa-lightbulb"
                    bounce
                    size="2xl"
                    class="mb-4"
                  />
                  <p>Thinking...</p>
                </div>

                <!-- Model Loading Animation -->
                <div
                  v-if="!selectedMessagePair.assistant"
                  class="flex-1 text-neutral-content flex flex-col items-center justify-center text-center"
                >
                  <font-awesome-icon icon="fa-solid fa-database" bounce size="2xl" class="mb-4" />
                  <p>Loading model...</p>
                  <p></p>
                  <p>Tip: Due to limited resource, sometimes you</p>
                  <p>may need to resend your description.</p>
                </div>

                <!-- Chip Code -->
                <div
                  v-if="
                    selectedMessagePair.assistant &&
                    !(
                      selectedMessagePair.assistant.isThinking &&
                      !selectedMessagePair.assistant.isOptimizing
                    )
                  "
                  class="flex-1 overflow-y-auto text-neutral-content text-sm px-4 py-2 bg-neutral"
                  ref="chatContainer"
                  :class="[
                    selectedMessagePair.assistant && selectedMessagePair.assistant.isOptimizing
                      ? 'bg-neutral-content bg-opacity-10 animate-pulse'
                      : '',
                  ]"
                >
                  <pre
                    v-for="(line, lineIndex) in trimmedLines"
                    :key="lineIndex"
                    :data-prefix="lineIndex + 1"
                    class="flex before:content-[attr(data-prefix)] pr-5 before:opacity-50 before:w-8 before:text-right before:mr-[2ch]"
                  >
                      <code v-html="highlightCode(line)"></code>
                  </pre>
                </div>

                <!-- Progress Bar -->
                <div class="h-[5px] overflow-hidden absolute bottom-0 w-full">
                  <div
                    v-if="
                      !selectedMessagePair.assistant || selectedMessagePair.assistant.isGenerating
                    "
                    class="bg-primary h-full w-full animate-loading origin-left"
                  ></div>
                  <div
                    v-if="
                      selectedMessagePair.assistant && selectedMessagePair.assistant.isOptimizing
                    "
                    class="bg-primary h-full transition-all"
                    :style="{ width: `${optimizationProgress}%` }"
                  ></div>
                </div>
              </div>

              <!-- Mobile slide to SVG button -->
              <button
                v-if="selectedMessagePair.assistant"
                @click="selectedMessagePair.assistant.activePanel = 'svg'"
                class="md:hidden absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 w-8 h-8 rounded-full bg-primary text-primary-content flex items-center justify-center shadow-md shadow-base-100 z-10"
              >
                <font-awesome-icon icon="fa-solid fa-angle-right" />
              </button>
            </div>

            <!-- Chip SVG -->
            <div
              class="flex w-full h-full shrink-0 md:shrink ml-[10vw] md:ml-4 relative transition-all duration-300"
            >
              <div
                class="flex w-full h-full overflow-hidden bg-neutral rounded-xl items-center justify-center relative"
              >
                <!-- Loading State -->
                <div
                  v-if="
                    !selectedMessagePair.assistant ||
                    selectedMessagePair.assistant.isGenerating ||
                    selectedMessagePair.assistant.isOptimizing ||
                    selectedMessagePair.assistant.isSynthesizing
                  "
                  class="text-neutral-content flex flex-col items-center text-center"
                >
                  <font-awesome-icon
                    icon="fa-solid fa-flask"
                    size="2xl"
                    class="mb-4 animate-[fa-spin_infinite_2s]"
                  />
                  <p
                    v-if="
                      !selectedMessagePair.assistant || selectedMessagePair.assistant.isGenerating
                    "
                  >
                    Generating microfluidic chip description...
                  </p>
                  <p v-else-if="selectedMessagePair.assistant.isOptimizing">
                    Trying to fix faulty chip design...
                  </p>
                  <p v-else-if="selectedMessagePair.assistant.isSynthesizing">
                    Synthesizing microfluidic chip...
                  </p>
                </div>

                <!-- Problem State -->
                <div
                  v-else-if="
                    (!selectedMessagePair.assistant.isOptimizing &&
                      selectedMessagePair.assistant.problems?.length) ||
                    (!selectedMessagePair.assistant.isSynthesizing &&
                      !selectedMessagePair.assistant.chipSvg)
                  "
                  class="text-neutral-content flex flex-col items-center text-center"
                >
                  <font-awesome-icon
                    icon="fa-solid fa-circle-exclamation"
                    size="2xl"
                    class="mb-4 text-error"
                  />

                  <template
                    v-if="
                      !selectedMessagePair.assistant.isOptimizing &&
                      selectedMessagePair.assistant.problems?.length
                    "
                  >
                    <p>Problems encountered during chip design.</p>
                    <p class="text-sm opacity-75">
                      Try prompting again to refine the faulty design and fix the issues.
                    </p>
                  </template>

                  <template v-else>
                    <p>Chip synthesis failed.</p>
                    <p class="text-sm text-stone-200 w-72">We provide only limited computational resources for testing purposes. For full performance and unrestricted synthesis capabilities, please deploy our platform on your own server.</p>
                  </template>
                </div>

                <!-- SVG Display -->
                <div v-else class="w-full h-full flex items-center justify-center">
                  <svg
                    v-if="selectedMessagePair.assistant.chipSvg"
                    v-html="selectedMessagePair.assistant.chipSvg"
                    class="w-full h-full"
                  ></svg>

                  <!-- SVG / STL Download Button in top-right corner -->
                  <div
                    v-if="selectedMessagePair.assistant.chipSvg"
                    class="absolute top-2 right-2 z-10 dropdown dropdown-hover dropdown-bottom dropdown-end"
                  >
                    <div
                      tabindex="0"
                      role="button"
                      class="btn btn-ghost text-base-content hover:text-primary"
                    >
                      <font-awesome-icon icon="fa-solid fa-download" size="lg" />
                    </div>
                    <ul
                      tabindex="0"
                      class="dropdown-content menu bg-base-100 rounded-box z-[1] w-52 p-2 shadow"
                    >
                      <li><a @click="downloadSVG">Download SVG (2D)</a></li>
                      <li
                        :class="{
                          'tooltip tooltip-bottom tooltip-error':
                            !selectedMessagePair.assistant.stlBase64,
                        }"
                        data-tip="STL generation timed out"
                      >
                        <a
                          @click="downloadSTL"
                          :class="{
                            'pointer-events-none opacity-50 cursor-not-allowed':
                              !selectedMessagePair.assistant.stlBase64,
                          }"
                          >Download STL (3D)</a
                        >
                      </li>
                    </ul>
                  </div>
                </div>

                <!-- Progress Bar -->
                <div
                  v-if="selectedMessagePair.assistant?.isSynthesizing"
                  class="h-[5px] overflow-hidden absolute bottom-0 w-full"
                >
                  <div class="bg-primary h-full w-full animate-loading origin-left"></div>
                </div>
              </div>

              <!-- Mobile slide to Code button -->
              <button
                v-if="selectedMessagePair.assistant"
                @click="selectedMessagePair.assistant.activePanel = 'code'"
                class="md:hidden absolute left-0 top-1/2 -translate-y-1/2 -translate-x-1/2 w-8 h-8 rounded-full bg-primary text-primary-content flex items-center justify-center shadow-md shadow-base-100 z-10"
              >
                <font-awesome-icon icon="fa-solid fa-angle-left" />
              </button>
            </div>
          </div>

          <!-- Chat Messages Container -->
          <div
            v-else
            ref="chatContainer"
            class="flex flex-col items-center grow w-full overflow-y-auto bg-stone-600 rounded-xl"
          >
            <div class="h-fit w-[90%] max-w-2xl p-4 my-2">
              <div
                v-for="(message, index) in chatMessages.filter((message) => !message.hidden)"
                :key="index"
                :class="[
                  'flex',
                  message.role === 'user' ? 'justify-end mb-2' : 'justify-start mb-8',
                ]"
              >
                <!-- Message -->
                <div
                  @click="
                    isMessageHoverable(message) ? selectMessagePair(message.messagePairId) : null
                  "
                  class="relative px-4 py-2 rounded-xl whitespace-break-spaces overflow-hidden"
                  :class="[
                    message.role === 'user'
                      ? 'max-w-[70%] bg-primary text-white text-sm'
                      : 'text-xs text-base-content',
                    isMessageHoverable(message)
                      ? 'hover:bg-neutral transition-colors duration-300 cursor-pointer'
                      : '',
                    message.isOptimizing && message.role === 'assistant'
                      ? 'bg-neutral animate-pulse'
                      : '',
                  ]"
                >
                  <span v-if="message.role !== 'assistant'" v-html="message.content"></span>
                  <pre v-else><code v-html="highlightCode(message.content)"></code></pre>

                  <!-- Progress Bar -->
                  <div
                    v-if="message.role === 'assistant' && message.isOptimizing"
                    class="h-[5px] overflow-hidden absolute bottom-0 right-0 left-0 w-full"
                  >
                    <div
                      class="bg-primary h-full transition-all"
                      :style="{ width: `${optimizationProgress}%` }"
                    ></div>
                  </div>
                </div>

                <!-- Problem info menu -->
                <div
                  v-if="
                    !message.isGenerating &&
                    !message.isOptimizing &&
                    message.role === 'assistant' &&
                    message.problems?.length
                  "
                  class="flex-1 relative min-w-40 min-h-60 items-center self-end"
                >
                  <div
                    @mouseleave="message.problemsHovered = false"
                    @mouseover="message.problemsHovered = true"
                    class="absolute bottom-0 left-0 mx-4 w-[50px] h-[50px] bg-error rounded-[25px] flex items-center justify-center transition-all duration-300 hover:w-40 hover:h-60 hover:rounded-xl overflow-x-hidden overflow-y-auto"
                  >
                    <button
                      v-if="!message.problemsHovered"
                      class="text-error-content cursor-default"
                    >
                      !
                    </button>
                    <div v-else class="flex flex-col items-center w-full h-full text-error-content">
                      <div class="px-4 py-2 text-xs font-bold">Problems</div>
                      <div
                        v-for="(problem, pIndex) in message.problems"
                        :key="pIndex"
                        class="w-40 px-4 py-2 text-xs"
                      >
                        {{ pIndex + 1 }}. {{ problem }}
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div
                v-if="
                  (isLoading &&
                    chatMessages.length > 0 &&
                    chatMessages[chatMessages.length - 1].role != 'assistant') ||
                  (chatMessages.length > 0 && chatMessages[chatMessages.length - 1].isGenerating)
                "
                class="flex flex-col justify-start px-4 items-start space-y-1"
              >
                <p
                  class="text-base-content text-sm"
                  v-if="
                    chatMessages[chatMessages.length - 1].isThinking &&
                    !chatMessages[chatMessages.length - 1].isOptimizing
                  "
                >
                  Thinking...
                </p>
                <div class="flex space-x-1">
                  <div
                    class="w-1.5 h-1.5 bg-base-content rounded-full animate-[bounce_1s_infinite_0ms]"
                  ></div>
                  <div
                    class="w-1.5 h-1.5 bg-base-content rounded-full animate-[bounce_1s_infinite_200ms]"
                  ></div>
                  <div
                    class="w-1.5 h-1.5 bg-base-content rounded-full animate-[bounce_1s_infinite_400ms]"
                  ></div>
                </div>
              </div>
            </div>
          </div>
        </div>

        <!-- Message Input Box -->
        <div class="flex flex-col mb-12 w-[90%] max-w-3xl bg-neutral rounded-2xl p-2">
          <textarea
            v-model="userMessage"
            :disabled="isLoading"
            placeholder="Describe any changes you want to make..."
            @keydown.enter.exact.prevent="handleSendClick"
            @input="adjustHeight"
            class="p-2 border-none bg-transparent text-neutral-content text-base resize-none outline-none h-[2.5rem] max-h-[110px] overflow-y-auto"
            ref="textarea"
          ></textarea>

          <div class="flex justify-between items-center mt-2">
            <div
              class="self-end"
              :class="[isLoading ? '' : 'tooltip tooltip-secondary tooltip-right']"
              :data-tip="[reasoningEnabled ? 'Disable reasoning' : 'Enable reasoning']"
            >
              <button
                @click="toggleReasoning"
                class="btn btn-circle btn-ghost btn-sm"
                :class="[
                  reasoningEnabled ? 'text-pink-300' : 'text-base-content',
                  isLoading ? 'pointer-events-none cursor-default' : '',
                ]"
              >
                <font-awesome-icon icon="fa-regular fa-lightbulb" size="lg" />
              </button>
            </div>

            <button
              @click="handleSendClick"
              :disabled="isLoading || !userMessage"
              class="w-[35px] h-[35px] rounded-full bg-cyan-400 text-neutral text-2xl font-bold flex justify-center items-center transition-colors duration-500 ease-in-out disabled:bg-neutral-content"
            >
              <span v-if="!isLoading">
                <font-awesome-icon icon="fa-solid fa-arrow-right" size="xs" />
              </span>
              <span
                v-else
                class="w-4 h-4 border-2 border-neutral border-t-transparent rounded-full animate-spin"
              ></span>
            </button>
          </div>
        </div>
      </div>
    </TransitionGroup>

    <!-- Reset Conversation Modal -->
    <dialog id="reset_conversation_modal" class="modal">
      <div class="modal-box">
        <form method="dialog">
          <button class="btn btn-sm btn-circle btn-ghost absolute right-2 top-2">✕</button>
        </form>
        <h3 class="text-lg font-bold">Reset Conversation?</h3>
        <p class="py-4">
          Are you sure you want to reset the conversation? This will erase all messages, and the
          chat cannot be restored.
        </p>
        <div class="modal-action">
          <form method="dialog">
            <button class="btn btn-ghost text-error" @click="resetChat">Reset</button>
          </form>
        </div>
      </div>
      <form method="dialog" class="modal-backdrop">
        <button class="outline-none">close</button>
      </form>
    </dialog>

    <!-- Discard Later Messages Modal -->
    <dialog id="discard_later_messages_modal" class="modal">
      <div class="modal-box">
        <form method="dialog">
          <button class="btn btn-sm btn-circle btn-ghost absolute right-2 top-2">✕</button>
        </form>
        <h3 class="text-lg font-bold">Discard Later Messages?</h3>
        <p class="py-4">
          You're about to send a message from an earlier point in the conversation. This will remove
          all messages that follow in the current context. Do you want to continue?
        </p>
        <div class="modal-action">
          <form method="dialog">
            <button class="btn btn-ghost text-primary" @click="discardLaterMessagesSendNew">
              Send
            </button>
          </form>
        </div>
      </div>
      <form method="dialog" class="modal-backdrop">
        <button class="outline-none">close</button>
      </form>
    </dialog>
  </div>
  <footer class="fixed bottom-0 left-0 z-20 w-full bg-gray-100 text-gray-700 text-sm text-center py-1 mt-8">
    <p>
      © 2025 EDA @ Technical University of Munich |
      <a href="https://www.ce.cit.tum.de/eda/datenschutz/" class="text-gray-500 hover:underline">
        Datenschutz
      </a>
      |
      <a href="https://www.ce.cit.tum.de/eda/impressum/" class="text-gray-500 hover:underline">
        Impressum
      </a>
    </p>
  </footer>
</template>

<script lang="ts">
import {
  type Ref,
  ref,
  useTemplateRef,
  type ShallowRef,
  onUnmounted,
  computed,
  nextTick,
  watch,
} from 'vue'
import { useToast } from 'vue-toastification'
import {
  AbortableAsyncIterator,
  parseJSON,
  type ChatResponse,
  type ErrorResponse,
} from '@/utils/chat-stream-utils'
import 'highlight.js/styles/vs2015.css'
import hljs from 'highlight.js/lib/core'
import json from 'highlight.js/lib/languages/json'

interface BaseMessage {
  content: string
  messagePairId: number
}

interface UserMessage extends BaseMessage {
  role: 'user'
  isThinking?: never
  isGenerating?: never
  isOptimizing?: never
  isSynthesizing?: never
  hidden?: boolean
  problems?: never
  problemsHovered?: never
  chipSvg?: never
  stlBase64?: never
  activePanel?: never
}

interface AssistantMessage extends BaseMessage {
  role: 'assistant'
  isThinking: boolean
  isGenerating: boolean
  isOptimizing: boolean
  isSynthesizing: boolean
  hidden?: never
  problems: Array<string>
  problemsHovered?: boolean
  chipSvg?: string
  stlBase64?: string
  activePanel: 'code' | 'svg'
}

type ChatMessage = UserMessage | AssistantMessage

export default {
  name: 'Flui3dChat',
  setup() {
    hljs.registerLanguage('json', json)

    const apiUrl = import.meta.env.VITE_API_URL
    const chatMessages = ref<ChatMessage[]>([])
    const messagePairIdCounter = ref(0)
    const userMessage = ref('')
    const isLoading = ref(false)
    const reasoningEnabled = ref(false)
    const toast = useToast()
    const chatContainer = useTemplateRef('chatContainer')
    const selectedMessagePairId = ref<number>(0)
    const selectedView = ref<number>(0)
    const currentTime = ref(Date.now())
    const settings = ref({
      chipMargin: { label: 'Chip Margin', value: 600 },
      layerHeight: { label: 'Layer Height', value: 200 },
      moduleMargin: { label: 'Module Margin', value: 400 },
      channelWidth: { label: 'Channel Width', value: 200 },
      channelMargin: { label: 'Channel Margin', value: 200 },
      portDiameter: { label: 'Port Diameter', value: 400 },
      serpentineWidth: { label: 'Serpentine Channel Width', value: 4000 },
      filterWidth: { label: 'Filter Width', value: 4000 },
      filterHeight: { label: 'Filter Height', value: 1200 },
      filterPillarRadius: { label: 'Filter Pillar Radius', value: 10 },
      maxRetries: { label: 'Maximum Retry Attempts', value: 3 },
    })

    const enforceInteger = (obj: { [key: string]: { value: number } }) => {
      Object.keys(obj).forEach((key) => {
        obj[key].value = Math.max(1, Math.round(Number(obj[key].value)) || 0)
      })
    }

    watch(settings, enforceInteger, { deep: true })

    let initialTime = 0
    let elapsedTime = 1
    let abortController: AbortController | null = null
    let intervalId: number | null = null

    onUnmounted(() => {
      if (abortController) {
        abortController.abort()
      }

      if (intervalId) {
        clearInterval(intervalId)
      }
    })

    const optimizationProgress = computed(() => {
      return (
        Math.round(
          Math.min(1, Math.max(0, (currentTime.value - initialTime) / elapsedTime)) * 1000,
        ) / 10
      )
    })

    const trimmedLines = computed(() => {
      if (selectedMessagePair.value && selectedMessagePair.value.assistant) {
        const lines = selectedMessagePair.value.assistant.content.split('\n')
        // Remove trailing empty lines
        while (lines.length && lines[lines.length - 1].trim() === '') {
          lines.pop()
        }
        return lines
      }
    })

    const shouldShowSendMessageModal = computed<boolean>(
      () => selectedView.value === 0 && selectedMessagePairId.value !== messagePairIdCounter.value,
    )

    // Shows the send message modal if messages would be discarded
    const showSendMessageModal = () => {
      const discardLaterMessagesModal = document.getElementById('discard_later_messages_modal')
      if (discardLaterMessagesModal && discardLaterMessagesModal instanceof HTMLDialogElement) {
        discardLaterMessagesModal.showModal()
      }
    }

    const handleSendClick = () => {
      if (isLoading.value) return
      if (!userMessage.value.trim()) return

      if (shouldShowSendMessageModal.value) {
        showSendMessageModal()
      } else {
        sendMessage()
      }
    }

    const discardLaterMessagesSendNew = () => {
      // Remove all messages with a messagePairId greater than selectedMessagePairId
      chatMessages.value = chatMessages.value.filter(
        (message) => message.messagePairId <= selectedMessagePairId.value,
      )

      // Reset the messagePairId counter
      messagePairIdCounter.value = selectedMessagePairId.value

      // Now send the new message
      sendMessage()
    }

    // Method to highlight code
    const highlightCode = (line: string) => {
      return hljs.highlight(line, {
        language: 'json',
        ignoreIllegals: false,
      }).value
    }

    // Computed property to safely access selected message pair
    const selectedMessagePair = computed(() => {
      // Find the user message with hidden = false and matching messagePairId
      const selectedUserMessage = chatMessages.value.find(
        (message) =>
          message.role === 'user' &&
          !message.hidden &&
          message.messagePairId === selectedMessagePairId.value,
      )

      if (selectedUserMessage) {
        // Find the corresponding assistant message with matching messagePairId

        const selectedAssistantMessage = chatMessages.value.find(
          (message) =>
            message.role === 'assistant' && message.messagePairId === selectedMessagePairId.value,
        )

        return {
          user: selectedUserMessage,
          assistant: selectedAssistantMessage ? selectedAssistantMessage : null,
        }
      }

      return null
    })

    // Function to select a message
    const selectMessagePair = (messagePairId: number) => {
      if (messagePairId >= 0) {
        selectedMessagePairId.value = messagePairId
      }

      // Switch to detail view
      selectedView.value = 0

      // Scroll to bottom for generating/optimizing message
      autoScroll(chatContainer, true)
    }

    // Function to switch back to chat view
    const switchView = () => {
      // Last message pair
      selectedMessagePairId.value = messagePairIdCounter.value

      selectedView.value = selectedView.value === 0 ? 1 : 0

      // Scroll to bottom for generating/optimizing message
      autoScroll(chatContainer, true)
    }

    const isMessageHoverable = (message: ChatMessage) => {
      return message.role === 'assistant' && !message.isGenerating && !message.isOptimizing
    }

    const sendMessage = async () => {
      if (isLoading.value) return
      if (!userMessage.value.trim()) return

      // Increment the messagePairId counter
      messagePairIdCounter.value += 1

      // setup AbortController
      abortController = new AbortController()

      chatMessages.value.push({
        role: 'user',
        content: userMessage.value,
        messagePairId: messagePairIdCounter.value,
      })
      userMessage.value = ''
      isLoading.value = true

      selectedMessagePairId.value = messagePairIdCounter.value
      if (selectedView.value === 0) {
        selectMessagePair(messagePairIdCounter.value)
      }

      adjustHeight()

      const assistantMessage = ref<ChatMessage>({
        role: 'assistant',
        content: '',
        isThinking: false,
        isGenerating: true,
        isOptimizing: false,
        isSynthesizing: false,
        problems: [],
        problemsHovered: false,
        messagePairId: messagePairIdCounter.value,
        activePanel: 'code',
      })
      let retries = 0

      try {
        while (retries <= settings.value.maxRetries.value) {
          // Measure time (in milliseconds)
          initialTime = Date.now()

          const response = await fetch(`${apiUrl}/api/chat`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              messages: chatMessages.value.map(({ role, content }) => ({ role, content })),
              reasoning: reasoningEnabled.value,
            }),
            signal: abortController.signal,
          })

          // Remove the hidden user message after retry
          chatMessages.value = chatMessages.value.filter((message) => !message.hidden)

          if (!response.ok || !response.body) {
            throw new Error('Failed to connect to backend.')
          }

          const itr = parseJSON<ChatResponse | ErrorResponse>(response.body)

          const abortableAsyncIterator = new AbortableAsyncIterator(abortController, itr, () => {
            // Reset abort controller
            abortController = new AbortController()
          })

          if (retries === 0) {
            // Add the single assistant message for updates
            chatMessages.value.push(assistantMessage.value)

            // Scroll to bottom if new message sent
            autoScroll(chatContainer, true)
          }

          let lineIndex = 0
          let currentPartialLine = ''
          let hasCompleteLine = false
          let newMessage = ''
          let buffer = ''
          let scrolled = false
          let inThinkBlock = false
          let thinkContent = ''
          let consumingPostThinkNewlines = false

          for await (const part of abortableAsyncIterator) {
            buffer += part.message.content
            let processableBuffer = ''

            // Loop to process the buffer for tags and extract actual content.
            while (true) {
              if (buffer.length === 0) {
                break // Buffer is empty, need more data from the stream
              }
              if (consumingPostThinkNewlines) {
                if (buffer[0] === '\n') {
                  buffer = buffer.substring(1) // Consume the newline
                  continue // Re-evaluate the modified buffer from the start of the while loop.
                } else {
                  // First character is not a newline. Stop consuming newlines.
                  consumingPostThinkNewlines = false
                }
              }
              if (!inThinkBlock) {
                const thinkStartIndex = buffer.indexOf('<think>')

                if (thinkStartIndex !== -1) {
                  buffer = buffer.substring(thinkStartIndex + '<think>'.length)
                  inThinkBlock = true
                } else {
                  // No <think> tag found in the rest of the buffer.
                  processableBuffer += buffer // All remaining buffer is actual content.
                  buffer = ''
                  break // Exit while loop, buffer is consumed for this part.
                }
              }

              if (inThinkBlock) {
                const thinkEndIndex = buffer.indexOf('</think>')
                if (thinkEndIndex !== -1) {
                  // Found the end of the think block.
                  thinkContent += buffer.substring(0, thinkEndIndex)
                  // The content of the think block is now in thinkContent.
                  // It will be disregarded for display.
                  // Set isThinking to false as the think block has ended.
                  assistantMessage.value.isThinking = false

                  buffer = buffer.substring(thinkEndIndex + '</think>'.length)
                  inThinkBlock = false
                  consumingPostThinkNewlines = true
                } else {
                  // Still inside the think block.
                  thinkContent += buffer // Accumulate content of the think block.
                  buffer = '' // All current buffer was part of the think block.

                  // If not already thinking and the accumulated think content is non-empty (trimmed),
                  // set isThinking to true.
                  if (!assistantMessage.value.isThinking && thinkContent.trim() !== '') {
                    assistantMessage.value.isThinking = true
                  }
                  break // Exit while loop, wait for more data for this think block.
                }
              }
            }

            if (processableBuffer.length > 0) {
              newMessage += processableBuffer
              const lines = processableBuffer.split('\n')

              // Handle the first incoming segment
              currentPartialLine += lines[0]
              const existingLines = assistantMessage.value.content.split('\n')

              // Check if we received a complete line (indicated by newline character)
              hasCompleteLine = lines.length > 1

              if (hasCompleteLine) {
                // Update current line and remove previous content only when moving to next line
                if (lineIndex < existingLines.length) {
                  existingLines[lineIndex] = currentPartialLine
                } else {
                  existingLines.push(currentPartialLine)
                }

                // Process additional complete lines
                for (let i = 1; i < lines.length; i++) {
                  lineIndex++
                  currentPartialLine = lines[i]

                  if (lineIndex < existingLines.length) {
                    // Preserve existing content for partial lines at the end
                    if (i === lines.length - 1) {
                      existingLines[lineIndex] =
                        existingLines[lineIndex].slice(0, currentPartialLine.length) +
                        existingLines[lineIndex].slice(currentPartialLine.length)
                    } else {
                      existingLines[lineIndex] = currentPartialLine
                    }
                  } else {
                    existingLines.push(currentPartialLine)
                  }
                }
              } else {
                // Update just the beginning portion of the current line, preserving the rest
                if (lineIndex < existingLines.length) {
                  existingLines[lineIndex] =
                    currentPartialLine + existingLines[lineIndex].slice(currentPartialLine.length)
                } else {
                  existingLines.push(currentPartialLine)
                }
              }

              assistantMessage.value.content = existingLines.join('\n')

              // autoscroll to bottom
              scrolled = autoScroll(chatContainer)
            }
          }

          // Clean up once streaming is complete
          assistantMessage.value.content = newMessage

          // Validate the assistant's response
          const validationResponse = await fetch(`${apiUrl}/api/validate`, {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify({
              chipDesign: JSON.parse(assistantMessage.value.content),
              fixJunctions: retries >= settings.value.maxRetries.value - 1,
            }),
          })

          if (!validationResponse.ok) {
            throw new Error('Validation API call failed.')
          }

          const { correctedChipDesign, problems, addedJunctions } = await validationResponse.json()

          assistantMessage.value.problems = problems

          // Replace the assistant message with the corrected design
          if (correctedChipDesign) {
            assistantMessage.value.content = JSON.stringify(correctedChipDesign, null, 2)

            // Autoscroll to bottom after correction
            if (scrolled) {
              autoScroll(chatContainer, true)
            } else {
              autoScroll(chatContainer, false)
            }
          }

          assistantMessage.value.isGenerating = false
          toast.dismiss('optimization_info_toast')

          if (
            (problems.length > 0 || addedJunctions.length > 0) &&
            retries <= settings.value.maxRetries.value - 1
          ) {
            // create interval for optimization progress bar
            if (intervalId) {
              clearInterval(intervalId)
            }
            intervalId = setInterval(() => {
              currentTime.value = Date.now() // Update reactive time
            }, 100) // Update every 100ms for smooth changes

            // Inform about optimization
            elapsedTime = Date.now() - initialTime // Time elapsed in ms
            if (problems.length > 0) {
              toast.info(
                `Mistakes in the generated chip design were detected and they are trying to be fixed. Attempt #${retries + 1}`,
                {
                  id: 'optimization_info_toast',
                },
              )

              // Add a hidden user message for problems
              const hiddenUserMessage: ChatMessage = {
                role: 'user',
                content: `After validating your chip design, the following problems were found: ${problems.join('; ')}. Please fix them while keeping the original description of your chip and possible changes made by the user in mind. **IMPORTANT:** It is really important that you stick to the user's description of the microfluidic chip provided earlier!`,
                hidden: true,
                messagePairId: messagePairIdCounter.value,
              }

              chatMessages.value.push(hiddenUserMessage)
            } else if (addedJunctions.length > 0) {
              // After programmatically fixing the junctions, attempt once to let the LLM adjust possible user-defined junction types.
              // If this fails, simply use the programmatically added junctions as they are.

              toast.info(
                `Junction network has been adjusted. The model is now adapting instructions accordingly.`,
                { id: 'optimization_info_toast' },
              )

              const hiddenUserMessage: ChatMessage = {
                role: 'user',
                content: `Please verify that the junction types (Y-junction or T-junction) for the following junctions: ${addedJunctions.join(', ')} are appropriate. Please modify only the junctions section of the JSON, leaving the connections and component_params sections unchanged.`,
                hidden: true,
                messagePairId: messagePairIdCounter.value,
              }

              chatMessages.value.push(hiddenUserMessage)
            }

            assistantMessage.value.isOptimizing = true
          } else {
            assistantMessage.value.isOptimizing = false

            // Chip is correct --> synthesis and generate svg/stl
            synthesizeChip(assistantMessage)

            break
          }

          retries++
        }

        assistantMessage.value.isOptimizing = false

        if (retries > settings.value.maxRetries.value) {
          toast.error('Error: Maximum validation attempts reached.')

          // Remove the hidden user messages
          chatMessages.value = chatMessages.value.filter((message) => !message.hidden)
        }
      } catch (error) {
        if (error instanceof Error && error.name !== 'AbortError') {
          toast.error('Error: Unable to connect to backend.\nTip: If the model just loaded, please resend your description.')
        }

        // Remove the hidden user messages
        chatMessages.value = chatMessages.value.filter((message) => !message.hidden)

        // Remove the assistant messages at the end if generation already started
        if (
          chatMessages.value.length > 0 &&
          chatMessages.value[chatMessages.value.length - 1].role === 'assistant'
        ) {
          chatMessages.value.pop()
        }

        // Remove the user message in case of failure
        chatMessages.value.pop()

        // Decrement the messagePairId counter
        messagePairIdCounter.value -= 1

        // Reset selected message pair
        selectedMessagePairId.value = messagePairIdCounter.value
      } finally {
        toast.dismiss('optimization_info_toast')

        // Remove the hidden user messages
        chatMessages.value = chatMessages.value.filter((message) => !message.hidden)

        isLoading.value = false

        if (intervalId) {
          clearInterval(intervalId)
        }
      }
    }

    const resynthesizeChip = () => {
      // close menu
      if (document.activeElement instanceof HTMLElement) {
        document.activeElement.blur()
      }

      // resynthesize with new settings
      if (
        selectedMessagePair.value &&
        selectedMessagePair.value.assistant &&
        selectedMessagePair.value.assistant.content &&
        !selectedMessagePair.value.assistant.isGenerating &&
        !selectedMessagePair.value.assistant.isOptimizing
      ) {
        const assistantRef = ref(selectedMessagePair.value.assistant) as Ref<ChatMessage>
        synthesizeChip(assistantRef)
      }
    }

    const synthesizeChip = async (assistantMessage: Ref<ChatMessage>) => {
      if (assistantMessage.value.role !== 'assistant') return
      if (!assistantMessage.value.content) return
      if (assistantMessage.value.isOptimizing) return
      if (assistantMessage.value.isGenerating) return

      try {
        // Set loading state
        assistantMessage.value.isSynthesizing = true
        assistantMessage.value.activePanel = 'svg'

        // Make API call
        const response = await fetch(`${apiUrl}/api/synthesize`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            Accept: 'application/json',
          },
          body: JSON.stringify({
            chipDesign: JSON.parse(assistantMessage.value.content),
            settings: {
              chipMargin: settings.value.chipMargin.value,
              layerHeight: settings.value.layerHeight.value,
              moduleMargin: settings.value.moduleMargin.value,
              channelWidth: settings.value.channelWidth.value,
              channelMargin: settings.value.channelMargin.value,
              portDiameter: settings.value.portDiameter.value,
              serpentineWidth: settings.value.serpentineWidth.value,
              filterWidth: settings.value.filterWidth.value,
              filterHeight: settings.value.filterHeight.value,
              filterPillarRadius: settings.value.filterPillarRadius.value,
            },
          }),
        })

        // Check if response is okay
        if (!response.ok) {
          throw new Error('Synthesis API call failed.')
        }

        const json = await response.json()

        // Update colors and save the generated SVG
        const processedSvg = json.svg
          // Replace background color
          .replace(
            /(stroke|fill)\s*:\s*(rgb\(\s*128\s*,\s*128\s*,\s*128\s*\))/gi,
            '$1: oklch(var(--n))',
          )
          // Replace element color
          .replace(
            /(stroke|fill)\s*:\s*(rgb\(\s*255\s*,\s*255\s*,\s*0\s*\))/gi,
            '$1: oklch(var(--p))',
          )
          // Replace pillar color
          .replace(
            /(stroke|fill)\s*:\s*(rgb\(\s*192\s*,\s*192\s*,\s*192\s*\))/gi,
            '$1: oklch(var(--nc))',
          )
          // Replace text color
          .replace(/(stroke|fill)\s*:\s*(rgb\(\s*0\s*,\s*0\s*,\s*0\s*\))/gi, '$1: oklch(var(--pc))')

        assistantMessage.value.chipSvg = processedSvg

        // Save STL data (if available) for download later
        if (json.stlBase64) {
          assistantMessage.value.stlBase64 = json.stlBase64
        }
      } catch (error) {
        console.error('Error synthesizing chip:', error)
      } finally {
        // Reset loading state regardless of success/error
        assistantMessage.value.isSynthesizing = false
      }
    }

    const autoScroll = (
      container: Readonly<ShallowRef<unknown>>,
      force: boolean = false,
    ): boolean => {
      nextTick(() => {
        if (container.value instanceof HTMLDivElement) {
          // For View 0 dont scroll if other than selected message is generated/optimized
          if (
            selectedView.value !== 0 ||
            (selectedMessagePair.value &&
              (selectedMessagePair.value.assistant?.isGenerating ||
                selectedMessagePair.value.assistant?.isOptimizing))
          ) {
            // Get the current scroll position
            const scrollTop = container.value.scrollTop
            const scrollHeight = container.value.scrollHeight
            const clientHeight = container.value.clientHeight

            // If the user is near the bottom (within 100px of the end)
            if (scrollHeight - scrollTop - clientHeight < 100 || force) {
              // Auto-scroll to the bottom
              container.value.scrollTop = scrollHeight

              return true
            }
          }
        }
      })

      return false
    }

    const adjustHeight = () => {
      const textarea = document.querySelector('textarea')
      if (textarea) {
        if (!userMessage.value) {
          textarea.value = ''
        }
        textarea.style.height = '2.5rem' // Reset the height to default to shrink before expanding
        textarea.style.height = `${textarea.scrollHeight}px` // Adjust the height to fit the content
      }
    }

    const resetChat = () => {
      chatMessages.value = []
      userMessage.value = ''
      selectedMessagePairId.value = 0

      if (abortController) {
        abortController.abort()

        // reset abort controller
        abortController = null
      }
    }

    const toggleReasoning = () => {
      if (!isLoading.value) {
        reasoningEnabled.value = !reasoningEnabled.value
      }
    }

    const downloadSVG = () => {
      if (
        selectedMessagePair.value &&
        selectedMessagePair.value.assistant &&
        selectedMessagePair.value.assistant.chipSvg
      ) {
        let svg = selectedMessagePair.value.assistant.chipSvg

        // Create a temporary element to resolve CSS variables
        const tempEl = document.createElement('div')
        tempEl.style.display = 'none'
        document.body.appendChild(tempEl)

        // Helper function to resolve CSS var
        function resolveColor(variable: string): string {
          tempEl.style.color = `oklch(var(${variable}))`
          return getComputedStyle(tempEl).color
        }

        // Resolve and replace oklch(var(--xxx)) with actual color values
        svg = svg
          .replace(/oklch\(var\(--n\)\)/g, resolveColor('--n'))
          .replace(/oklch\(var\(--p\)\)/g, resolveColor('--p'))
          .replace(/oklch\(var\(--nc\)\)/g, resolveColor('--nc'))
          .replace(/oklch\(var\(--pc\)\)/g, resolveColor('--pc'))

        // Remove temp element
        document.body.removeChild(tempEl)

        const blob = new Blob([svg], { type: 'image/svg+xml' })
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'microfluidic_chip.svg'
        a.click()
        URL.revokeObjectURL(url)
      }
    }

    const downloadSTL = () => {
      if (
        selectedMessagePair.value &&
        selectedMessagePair.value.assistant &&
        selectedMessagePair.value.assistant.stlBase64
      ) {
        const blob = new Blob(
          [
            Uint8Array.from(atob(selectedMessagePair.value.assistant.stlBase64), (c) =>
              c.charCodeAt(0),
            ),
          ],
          { type: 'model/stl' },
        )
        const url = URL.createObjectURL(blob)
        const a = document.createElement('a')
        a.href = url
        a.download = 'microfluidic_chip.stl'
        a.click()
        URL.revokeObjectURL(url)
      }
    }

    return {
      chatMessages,
      userMessage,
      isLoading,
      sendMessage,
      resetChat,
      adjustHeight,
      autoScroll,
      selectedMessagePair,
      selectMessagePair,
      switchView,
      isMessageHoverable,
      highlightCode,
      selectedView,
      optimizationProgress,
      resynthesizeChip,
      synthesizeChip,
      handleSendClick,
      discardLaterMessagesSendNew,
      messagePairIdCounter,
      settings,
      reasoningEnabled,
      toggleReasoning,
      trimmedLines,
      downloadSVG,
      downloadSTL,
    }
  },
}
</script>

<style scoped>
.chat-transition-move,
.chat-transition-enter-active,
.chat-transition-leave-active {
  transition: all 0.5s cubic-bezier(0.4, 0, 0.2, 1);
}

.chat-transition-enter-from,
.chat-transition-leave-to {
  opacity: 0;
  transform: translateY(30px);
}

.chat-transition-leave-active {
  position: absolute;
}

@keyframes loading {
  0% {
    transform: translateX(0) scaleX(0);
  }

  20% {
    transform: translateX(0) scaleX(0.3);
  }

  100% {
    transform: translateX(100%) scaleX(0.4);
  }
}

.animate-loading {
  animation: loading 3.5s infinite linear;
}
</style>
