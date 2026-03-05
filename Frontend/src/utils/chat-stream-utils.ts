// This file contains code adapted from the Ollama JS project.
// https://github.com/ollama/ollama-js

// MIT License

// Copyright (c) 2023 Saul

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

export interface ErrorResponse {
  error: string
}

interface Message {
  role: string
  content: string
}

export interface ChatResponse {
  model: string
  created_at: Date
  message: Message
  done: boolean
  done_reason: string
  total_duration: number
  load_duration: number
  prompt_eval_count: number
  prompt_eval_duration: number
  eval_count: number
  eval_duration: number
}

/**
 * Parses a ReadableStream of Uint8Array into JSON objects.
 * @param itr {ReadableStream<Uint8Array>} - The stream to parse
 * @returns {AsyncGenerator<T>} - The parsed JSON objects
 */
export const parseJSON = async function* <T = unknown>(
  itr: ReadableStream<Uint8Array>,
): AsyncGenerator<T> {
  const decoder = new TextDecoder('utf-8')
  let buffer = ''

  const reader = itr.getReader()

  while (true) {
    const { done, value: chunk } = await reader.read()

    if (done) {
      break
    }

    buffer += decoder.decode(chunk)

    const parts = buffer.split('\n')

    buffer = parts.pop() ?? ''

    for (const part of parts) {
      try {
        yield JSON.parse(part)
      } catch (error) {
        console.warn('invalid json: ', part)
      }
    }
  }

  for (const part of buffer.split('\n').filter((p) => p !== '')) {
    try {
      yield JSON.parse(part)
    } catch (error) {
      console.warn('invalid json: ', part)
    }
  }
}

/**
 * An AsyncIterator which can be aborted
 */
export class AbortableAsyncIterator<T extends object> {
  private readonly abortController: AbortController
  private readonly itr: AsyncGenerator<T | ErrorResponse>
  private readonly doneCallback: () => void

  constructor(
    abortController: AbortController,
    itr: AsyncGenerator<T | ErrorResponse>,
    doneCallback: () => void,
  ) {
    this.abortController = abortController
    this.itr = itr
    this.doneCallback = doneCallback
  }

  abort() {
    this.abortController.abort()
  }

  async *[Symbol.asyncIterator]() {
    for await (const message of this.itr) {
      if ('error' in message) {
        throw new Error(message.error)
      }
      yield message
      // message will be done in the case of chat and generate
      // message will be success in the case of a progress response (pull, push, create)
      if ((message as any).done || (message as any).status === 'success') {
        this.doneCallback()
        return
      }
    }
    throw new Error('Did not receive done or success response in stream.')
  }
}
