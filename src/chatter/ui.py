"""
User interface components using Gradio.
"""

import time
from typing import Any

import gradio as gr
import numpy as np
from numpy.typing import NDArray

from .assistant import VoiceAssistant
from .config import Config


# Type definitions
GradioHistory = list[list[str]]
GradioUpdate = dict[str, Any]
AudioTuple = tuple[int, NDArray[np.float32]] | None


class VoiceChatInterface:
    """Enhanced ChatInterface with voice capabilities."""

    def __init__(self, assistant: VoiceAssistant):
        self.assistant = assistant

    def respond_to_message(self, message: str, _history: list[list[str]]) -> str:
        """Process a text message and return AI response."""
        # Add user message to conversation
        self.assistant.conversation.add_user_message(message)

        # Get LLM response
        llm_response, _ = self.assistant.llm.get_response(
            self.assistant.conversation.get_messages_for_llm()
        )

        if llm_response is None:
            return "Sorry, I encountered an error processing your message."

        # Add assistant message to conversation
        self.assistant.conversation.add_assistant_message(llm_response)

        # Generate TTS audio (for consistency, though not used in text interface)
        self.assistant.tts.synthesize(
            llm_response, self.assistant.conversation.get_current_persona()
        )

        # Return response as-is
        return llm_response

    def create_interface(self) -> gr.Blocks:
        """Create a custom interface that combines ChatInterface with voice controls."""
        # Import RecordingState here to fix the undefined name error
        from .assistant import RecordingState

        with gr.Blocks(
            title="Chatter",
            theme="soft",
            css="""
                .minimal-audio {
                    height: 30px !important;
                    opacity: 0.3;
                    transition: opacity 0.3s ease;
                }
                .minimal-audio:hover {
                    opacity: 1;
                }
                #tts-audio {
                    max-height: 40px !important;
                }
                .loading-spinner {
                    display: inline-block;
                    width: 20px;
                    height: 20px;
                    border: 3px solid #f3f3f3;
                    border-radius: 50%;
                    border-top: 3px solid #3498db;
                    animation: spin 1s linear infinite;
                    margin-right: 8px;
                }
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .processing-status {
                    display: flex;
                    align-items: center;
                    color: #666;
                    font-style: italic;
                    padding: 10px;
                    background-color: #f8f9fa;
                    border-radius: 8px;
                    margin: 5px 0;
                }
            """,
        ) as interface:
            gr.Markdown("# 💬 Chatter")
            gr.Markdown(
                "**Conversational AI with voice input - Use the recording button to talk**"
            )

            # Persona selection
            with gr.Row():
                persona_dropdown = gr.Dropdown(
                    choices=self.assistant.persona_manager.get_persona_names(),
                    value=self.assistant.conversation.get_current_persona(),
                    label="🎭 Select Persona",
                    info="Choose the AI assistant's personality and expertise",
                    scale=3,
                )
                chime_btn = gr.Button("🔔 Chime In", variant="primary", scale=1)

            # Main chat interface
            chatbot = gr.Chatbot(
                height=500,
                show_copy_button=True,
                bubble_full_width=False,
                render_markdown=True,
                latex_delimiters=[
                    {"left": "$$", "right": "$$", "display": True},
                    {"left": "$", "right": "$", "display": False},
                ],
            )

            # Message input (will be populated by voice or typed)
            msg_input = gr.Textbox(
                placeholder="Type your message or use voice input...",
                container=False,
                scale=4,
            )

            # Voice controls
            with gr.Row():
                start_btn = gr.Button("🎙️ Start Recording", variant="primary", size="lg")
                stop_btn = gr.Button(
                    "🔴 Stop & Transcribe", variant="stop", size="lg", visible=False
                )
                submit_btn = gr.Button("Send", variant="primary", size="lg")

            # Status display for processing feedback
            status_display = gr.HTML(
                value="", visible=False, elem_classes=["processing-status"]
            )

            # Audio for TTS (minimally visible for functionality)
            audio_output = gr.Audio(
                type="numpy",
                autoplay=True,
                visible=True,
                show_label=False,
                elem_id="tts-audio",
                elem_classes=["minimal-audio"],
            )

            # State management
            recording_state = gr.State(False)

            def create_loading_html(message: str) -> str:
                """Create HTML with loading spinner."""
                return f'<div class="processing-status"><div class="loading-spinner"></div>{message}</div>'

            def change_persona(persona_name: str) -> GradioUpdate:
                """Change the current persona."""
                self.assistant.conversation.set_persona(persona_name)
                return gr.update(
                    value=f"✅ Switched to {persona_name} persona", visible=True
                )

            def clear_chat_sync() -> tuple[GradioHistory, str, GradioUpdate]:
                """Sync conversation history when chatbot is cleared."""
                self.assistant.conversation.clear()
                return [], "", gr.update(value="", visible=False)

            # Hook into chatbot clear functionality
            chatbot.clear(
                fn=clear_chat_sync, outputs=[chatbot, msg_input, status_display]
            )

            def chime_in():
                """Get the current persona to chime in on the conversation."""
                if not self.assistant.conversation.history:
                    # No conversation yet, just start naturally
                    chime_prompt = "Start a conversation naturally from your perspective and expertise."
                else:
                    # Jump into the conversation naturally without meta-commentary
                    chime_prompt = "Continue this conversation naturally with your own thoughts, questions, or insights. Don't announce that you're chiming in - just contribute to the discussion as if you were already part of it."

                # Show chiming status
                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(
                        value=create_loading_html(
                            f"{self.assistant.conversation.get_current_persona()} is thinking..."
                        ),
                        visible=True,
                    ),
                    None,
                )

                # Add the chime prompt as a user message (but don't display it in chat)
                messages_for_llm = (
                    self.assistant.conversation.get_messages_for_llm()
                    + [{"role": "user", "content": chime_prompt}]
                )

                # Get streaming LLM response
                try:
                    accumulated_response = ""
                    current_history = self.assistant.conversation.get_chat_history()

                    # Add empty assistant message to show streaming
                    current_history.append(
                        [f"🤖 {self.assistant.conversation.get_current_persona()}", ""]
                    )

                    for (
                        chunk_content,
                        full_response,
                    ) in self.assistant.llm.get_streaming_response(messages_for_llm):
                        if chunk_content is None:  # Error case
                            yield (
                                current_history,
                                "",
                                gr.update(value=f"❌ {full_response}", visible=True),
                                None,
                            )
                            return

                        # Update the response in real-time
                        accumulated_response = full_response
                        cleaned_response = self.assistant.llm.parse_deepseek_response(
                            accumulated_response
                        )
                        current_history[-1][1] = cleaned_response
                        yield (
                            current_history,
                            "",
                            gr.update(
                                value=create_loading_html(
                                    f"{self.assistant.conversation.get_current_persona()} is responding..."
                                ),
                                visible=True,
                            ),
                            None,
                        )

                    # Clean the final response
                    final_response = self.assistant.llm.parse_deepseek_response(
                        accumulated_response
                    )
                    current_history[-1][1] = final_response

                    # Add assistant message to conversation (this will show in chat)
                    self.assistant.conversation.add_assistant_message(final_response)

                    # Show TTS generation status
                    yield (
                        self.assistant.conversation.get_chat_history(),
                        "",
                        gr.update(
                            value=create_loading_html("Generating speech..."),
                            visible=True,
                        ),
                        None,
                    )

                    # Generate TTS audio
                    audio_data, _ = self.assistant.tts.synthesize(
                        final_response,
                        self.assistant.conversation.get_current_persona(),
                    )
                    audio_output_data = (
                        (Config.TTS_SAMPLE_RATE, audio_data)
                        if audio_data is not None
                        else None
                    )

                except Exception as e:
                    yield (
                        self.assistant.conversation.get_chat_history(),
                        "",
                        gr.update(value=f"❌ Error: {str(e)}", visible=True),
                        None,
                    )
                    return

                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(value="✅ Chimed in!", visible=True),
                    audio_output_data,
                )

                # Hide status after a brief delay
                time.sleep(2)
                yield (
                    self.assistant.conversation.get_chat_history(),
                    "",
                    gr.update(value="", visible=False),
                    audio_output_data,
                )

            def start_recording():
                if self.assistant.state != RecordingState.IDLE:
                    return (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        "",
                        gr.update(value="", visible=False),
                    )

                self.assistant.state = RecordingState.RECORDING
                result = self.assistant.recorder.start_recording()
                print(f"Recording started: {result}")  # Debug
                return (
                    True,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    "",
                    gr.update(value=result, visible=True),
                )

            def stop_recording_and_transcribe():
                if self.assistant.state != RecordingState.RECORDING:
                    # Return generator for consistent interface
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(value="", visible=False),
                        None,
                    )
                    return

                self.assistant.state = RecordingState.PROCESSING

                # Show processing status
                yield (
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    chatbot.value,
                    "",
                    gr.update(
                        value=create_loading_html("Processing audio..."), visible=True
                    ),
                    None,
                )

                # Stop recording and get the transcribed text
                print(
                    f"Recording state before stop: {self.assistant.recorder.recording}"
                )
                print(
                    f"Audio chunks before stop: {len(self.assistant.recorder.audio_data) if self.assistant.recorder.audio_data else 0}"
                )
                audio_data = self.assistant.recorder.stop_recording()
                print(f"Audio data received: {audio_data is not None}")
                if audio_data is not None:
                    print(f"Audio data length: {len(audio_data)}")
                else:
                    print(
                        "❌ Audio data is None - check console for stop_recording debug output"
                    )

                if audio_data is None:
                    self.assistant.state = RecordingState.IDLE
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(value="No audio recorded", visible=True),
                        None,
                    )
                    return

                # Show transcription status
                yield (
                    False,
                    gr.update(visible=False),
                    gr.update(visible=True),
                    chatbot.value,
                    "",
                    gr.update(
                        value=create_loading_html("Transcribing speech..."),
                        visible=True,
                    ),
                    None,
                )

                # Transcribe the audio
                transcribed_text, transcription_status = (
                    self.assistant.transcription.transcribe(
                        audio_data, self.assistant.recorder.sample_rate
                    )
                )
                print(f"Transcription result: {transcribed_text}")  # Debug

                if transcribed_text is None:
                    self.assistant.state = RecordingState.IDLE
                    yield (
                        False,
                        gr.update(visible=True),
                        gr.update(visible=False),
                        chatbot.value,
                        "",
                        gr.update(
                            value=transcription_status or "Transcription failed",
                            visible=True,
                        ),
                        None,
                    )
                    return

                # Populate the input box with transcribed text for user review/editing
                self.assistant.state = RecordingState.IDLE
                yield (
                    False,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    chatbot.value,
                    transcribed_text,
                    gr.update(
                        value=f'✅ Transcribed: "{transcribed_text}" - Review and click Send',
                        visible=True,
                    ),
                    None,
                )

                # Hide status after a brief delay
                time.sleep(3)
                yield (
                    False,
                    gr.update(visible=True),
                    gr.update(visible=False),
                    chatbot.value,
                    transcribed_text,
                    gr.update(value="", visible=False),
                    None,
                )

            def process_message(message: str, history: list[list[str | None]]):
                if not message.strip():
                    return history, "", gr.update(value="", visible=False), None

                # Add user message to history
                history = history + [[message, None]]

                # Show AI processing status
                yield (
                    history,
                    "",
                    gr.update(
                        value=create_loading_html("AI is thinking..."), visible=True
                    ),
                    None,
                )

                # Add user message to conversation
                self.assistant.conversation.add_user_message(message)

                # Get streaming LLM response
                try:
                    accumulated_response = ""
                    for (
                        chunk_content,
                        full_response,
                    ) in self.assistant.llm.get_streaming_response(
                        self.assistant.conversation.get_messages_for_llm()
                    ):
                        if chunk_content is None:  # Error case
                            yield (
                                history,
                                "",
                                gr.update(value=f"❌ {full_response}", visible=True),
                                None,
                            )
                            return

                        # Update the response in real-time
                        accumulated_response = full_response
                        history[-1][1] = self.assistant.llm.parse_deepseek_response(
                            accumulated_response
                        )
                        yield (
                            history,
                            "",
                            gr.update(
                                value=create_loading_html("AI is responding..."),
                                visible=True,
                            ),
                            None,
                        )

                    # Clean the final response
                    final_response = self.assistant.llm.parse_deepseek_response(
                        accumulated_response
                    )
                    history[-1][1] = final_response

                    # Add assistant message to conversation
                    self.assistant.conversation.add_assistant_message(final_response)

                    # Show TTS generation status
                    yield (
                        history,
                        "",
                        gr.update(
                            value=create_loading_html("Generating speech..."),
                            visible=True,
                        ),
                        None,
                    )

                    # Generate TTS audio
                    audio_output_data = None
                    audio_data, _ = self.assistant.tts.synthesize(
                        final_response,
                        self.assistant.conversation.get_current_persona(),
                    )
                    if audio_data is not None:
                        audio_output_data = (Config.TTS_SAMPLE_RATE, audio_data)

                    yield (
                        history,
                        "",
                        gr.update(value="✅ Complete!", visible=True),
                        audio_output_data,
                    )

                    # Hide status after a brief delay
                    time.sleep(2)
                    yield (
                        history,
                        "",
                        gr.update(value="", visible=False),
                        audio_output_data,
                    )

                except Exception as e:
                    yield (
                        history,
                        "",
                        gr.update(value=f"❌ Error: {str(e)}", visible=True),
                        None,
                    )

            # Event handlers
            start_btn.click(
                fn=start_recording,
                outputs=[
                    recording_state,
                    start_btn,
                    stop_btn,
                    msg_input,
                    status_display,
                ],
            )

            stop_btn.click(
                fn=stop_recording_and_transcribe,
                outputs=[
                    recording_state,
                    start_btn,
                    stop_btn,
                    chatbot,
                    msg_input,
                    status_display,
                    audio_output,
                ],
            )

            submit_btn.click(
                fn=process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, status_display, audio_output],
            )

            msg_input.submit(
                fn=process_message,
                inputs=[msg_input, chatbot],
                outputs=[chatbot, msg_input, status_display, audio_output],
            )

            # Persona and clear chat handlers
            persona_dropdown.change(
                fn=change_persona, inputs=[persona_dropdown], outputs=[status_display]
            )

            chime_btn.click(
                fn=chime_in, outputs=[chatbot, msg_input, status_display, audio_output]
            )

            # Enhanced audio auto-play functionality
            interface.load(
                None,
                None,
                None,
                js="""
                function() {

                    // Enhanced audio auto-play
                    function tryPlayAudio(audioElement) {
                        if (audioElement && (audioElement.src || audioElement.srcObject)) {
                            console.log('Attempting to play audio...');
                            audioElement.play()
                                .then(() => {
                                    console.log('✅ Audio playing successfully');
                                })
                                .catch(e => {
                                    console.log('❌ Auto-play blocked:', e);
                                    // Try to enable audio on user interaction
                                    document.addEventListener('click', function enableAudio() {
                                        audioElement.play().catch(console.log);
                                        document.removeEventListener('click', enableAudio);
                                    }, { once: true });
                                });
                        }
                    }

                    // Monitor for audio updates
                    const observer = new MutationObserver(function(mutations) {
                        mutations.forEach(function(mutation) {
                            // Check for new audio elements
                            mutation.addedNodes.forEach(function(node) {
                                if (node.nodeType === 1) {
                                    const audioElements = node.querySelectorAll ? node.querySelectorAll('audio') : [];
                                    const directAudio = node.tagName === 'AUDIO' ? [node] : [];
                                    const allAudio = [...audioElements, ...directAudio];

                                    allAudio.forEach(audioElement => {
                                        setTimeout(() => tryPlayAudio(audioElement), 50);
                                    });
                                }
                            });

                            // Check for audio source changes
                            if (mutation.type === 'attributes' && mutation.target.tagName === 'AUDIO') {
                                setTimeout(() => tryPlayAudio(mutation.target), 50);
                            }
                        });
                    });

                    // Also check existing audio elements periodically
                    setInterval(() => {
                        const audioElements = document.querySelectorAll('audio');
                        audioElements.forEach(audio => {
                            if (audio.src && audio.paused && audio.readyState >= 2) {
                                tryPlayAudio(audio);
                            }
                        });
                    }, 500);

                    observer.observe(document.body, {
                        childList: true,
                        subtree: true,
                        attributes: true,
                        attributeFilter: ['src']
                    });
                }
                """,
            )

        return interface
