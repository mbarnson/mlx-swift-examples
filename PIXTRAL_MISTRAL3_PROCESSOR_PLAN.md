# Pixtral/Mistral3 Processor Implementation Plan

## Goal
Implement processors that maximize compatibility with mistral-common's approach while working within Swift/MLX ecosystem constraints.

## mistral-common Compatibility Points

### ✅ What We Align With:
1. **Chat Template Application** - via `swift-transformers`'s `applyChatTemplate()`
   - Uses same Jinja templates from tokenizer_config.json
   - Supports tools, system messages, multi-turn conversations
   - Handles special tokens ([INST], [IMG], etc.) correctly

2. **Message Protocol** - via structured `UserInput.Prompt.chat([Chat.Message])`
   - Equivalent to UserMessage/AssistantMessage in mistral-common
   - Supports role-based messaging (system, user, assistant, tool)
   - Handles image/video attachments

3. **Tokenization** - via HuggingFace tokenizer files
   - Same tokenizer.json and tokenizer_config.json
   - Tekken tokenizer for modern Mistral models
   - Same vocabulary and special tokens

### ⚠️ Where We Reasonably Diverge:
1. **No Pydantic Validation** - Swift doesn't have Pydantic
   - Acceptable: Type safety via Swift's type system
   - Validation happens at compile time

2. **No mistral-common Python bindings** - Swift is native
   - Acceptable: Can't call Python from production Swift apps
   - Pattern alignment is what matters

3. **Image preprocessing differences** - MLX vs torch
   - Acceptable: Different ML frameworks require different approaches
   - MLX native CIImage → MLXArray pipeline

## Implementation Architecture

### Pixtral Processor
```swift
public func prepare(input: UserInput) async throws -> LMInput {
    // 1. Message generation (mistral-common compatibility)
    let messageGenerator = PixtralMessageGenerator()
    let messages = messageGenerator.generate(from: input)

    // 2. Chat template application (CRITICAL mistral-common step)
    var promptTokens = try tokenizer.applyChatTemplate(
        messages: messages,
        tools: input.tools,
        additionalContext: input.additionalContext
    )

    // 3. Image preprocessing (MLX-specific but semantically equivalent)
    if !input.images.isEmpty {
        let pixelValues = try preprocessImages(input.images, processing: input.processing)
        return LMInput(tokens: MLXArray(promptTokens), image: pixelValues)
    }

    return LMInput(tokens: MLXArray(promptTokens))
}
```

### Mistral3 Processor (extends Pixtral)
```swift
public func prepare(input: UserInput) async throws -> LMInput {
    // Same as Pixtral, but adds image_sizes for spatial merging
    let lmInput = try await super.prepare(input: input)

    if let pixelValues = lmInput.image {
        let imageSizes = calculateImageSizes(from: input.images)
        return LMInput(
            tokens: lmInput.tokens,
            image: pixelValues,
            imageSizes: imageSizes
        )
    }

    return lmInput
}
```

## Image Preprocessing Logic

### Pixtral (Simple):
1. Load CIImage from UserInput.Image
2. Apply user-requested processing (resize if specified)
3. Convert to sRGB tone curve
4. Convert to MLXArray [1, C, H, W]
5. Concatenate multiple images along batch dimension

### Mistral3 (Adds Size Tracking):
1. Same as Pixtral
2. Additionally track original (h, w) for each image
3. Return image_sizes as MLXArray [[h1, w1], [h2, w2], ...]
4. Used by model for 2x2 spatial patch merging

## Message Generation

Both use same pattern:
```swift
class PixtralMessageGenerator: MessageGenerator {
    func generate(from input: UserInput) -> [Message] {
        switch input.prompt {
        case .chat(let messages):
            // Convert Chat.Message to [String: Any] format
            return messages.map { message in
                var dict: [String: Any] = ["role": message.role.rawValue]

                // Handle text + images
                if !message.images.isEmpty {
                    var content: [[String: Any]] = []
                    content.append(["type": "text", "text": message.content])
                    for _ in message.images {
                        content.append(["type": "image"])
                    }
                    dict["content"] = content
                } else {
                    dict["content"] = message.content
                }

                return dict
            }
        case .text(let text):
            return [["role": "user", "content": text]]
        case .messages(let messages):
            return messages
        }
    }
}
```

## Testing Strategy

1. **Unit Tests**: Message generation matches expected format
2. **Integration Tests**: Full pipeline with actual Magistral model
3. **Compatibility Tests**: Compare token sequences with Python mlx-vlm
4. **Real Usage**: Test in SillyVision app with actual conversations

## Why This Approach is Idiomatic

✅ **Chat template via tokenizer** - Exact same mechanism as mistral-common
✅ **Message structure** - Semantically equivalent to mistral-common protocol
✅ **Special token handling** - Tokenizer handles [IMG], [INST], etc.
✅ **Multi-modal support** - Images integrated via content array
✅ **Tool support** - Passed through to applyChatTemplate()

The key insight: **mistral-common is primarily about message formatting and tokenization, not low-level image preprocessing**. By correctly using `applyChatTemplate()` with proper message structures, we achieve full compatibility with Mistral's expected input format.
