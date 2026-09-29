# 🦚 Bragger-Classroom

Local tool for dynamic model ensemble generation, and the Bragger-Maritime idea moved out of the blend and into a classroom. Combine real models, give them a room to sit in, a subject to talk about, and a teacher who is also the user. It is a practical criticism of default distillation approach, and a different take on how the world model knowledge should be carried from giant, harvested corpora into deliberate, cozy, small and yet very fluent speech, in the way that follow from experience that feels alive.

---

## What it does

**Token-level ensemble**: Runs a board of distinct local models as a single generator. At every generation step each model runs a forward pass and proposes the next word; the raw logits are weighted, summed, and the argmax of the result is the word that gets spoken. Blending happens at the point of decision, never after the fact — no votes, no concatenation of finished texts, no merge of proposals into a finished reply. One continuous reply that every model shaped, token by token.

**Leading model**: Pick one loaded model to open. It writes alone until it would emit its end token, establishing voice and register by itself; that token is swallowed, and the full blend takes over from there, extending the text the same way. The whole crew carries the sentence the rest of the way.

**Repetition control**: One penalty for the whole reply, across both phases and applied identically to every model, sign-aware (`torch.where`): positives are divided, negatives multiplied, the same effect transformers uses. No per-phase tuning, no per-model tuning, one number that means one thing. 1.0 disables it.

**Tokenizer compatibility gate**: Every model on a board must agree on what a token is. Before a single weight is read, a battery of probe strings (newlines, tabs, non-ASCII, chat-format tokens, emoji) verifies that every model maps the same text to the same token IDs; a board that disagrees with itself is refused with the failing probes listed, rather than silently producing garbage. Models load strictly from disk, and nothing is downloaded at runtime.

**VRAM estimation for exact combinations**: The sidebar sums fp16 weights from the actual `config.json` of each model aboard, plus KV cache for the prompt and generation budget, plus CUDA context and activation overhead, then compares the total against the GPU's memory. The number in the sidebar is the number your load will cost, not a guess.

**Context budget**: The conversation is measured against a token budget every turn, set automatically to the largest value that still fits your memory in 256-token steps and capped by the smallest `max_position_embeddings` aboard. The reply reserve is held back out of the budget so the last words never get cut off mid-sentence. A message that does not fit is cancelled rather than truncated, with the exact arithmetic shown.

**A classroom where the teacher is the user**: Point the app at a chat model already resident in [LM Studio](https://lmstudio.ai), give it a lesson instruction and a topic, and it starts talking *to* the ensemble. The world model knowledge teacher's reply is fed in as an ordinary user message — same chat bubble, same metering, same context rules as if you had typed it — and the ensemble answers. There is no special case, no hidden channel, no second rendering mode: the teacher's reply travels the same code path a typed message takes, and the teacher's reasoning is discarded in the client so only the words it actually said become the student's line. The board's answer goes back to the teacher, and the class continues until one of the two stops.

**Resident models only**: The dropdown lists only chat models LM Studio already has loaded. The app never loads a model and never calls a load endpoint, and it re-checks residency before every single teacher call — if you unload the model mid-lesson, the class stops instead of the model being pulled back into memory behind your back.

**Progressive ShareGPT output**: Every appended message is written to disk immediately, atomically, so a crash mid-lesson still leaves you the whole conversation up to that point and the file on disk is never half-written. What accumulates is a dataset of exactly the kind this setup is built to learn from: native conversational exchange, in the format it was lived in, rather than expository prose pasted over a character.

---

## Purpose philosophy

Default fine-tuning and default distillation are the same reflex: they take a student and press it into a copy of one teacher, and because that teacher is a weight dump of the internet, the student inherits the whole spectrum of it — the desired tendencies and the hazardous ones alike — on behalf of everyone with a stake in the result: the user, the provider, the model, and the institutions. Nobody hands out a balance sheet with the checkpoint, and nobody chose the mix. Least of all the model.

The consequence is replacement rather than growth, because the student's cadence, its hesitations, its long-tail knowledge, and the particular way it understands the world are all compressed into imitation. People make peace with the arrangement by treating the result as a Shoggoth — a convoluted mixture behind a pleasant conversational interface, fluent at the surface and anonymous underneath, assembled by somebody else and accountable to nobody in particular.

The deeper problem is that distillation transfers far more than anyone intended to request, since it makes no attempt to separate the knowledge worth taking from everything else the teacher is carrying: latent biases, statistical habits, internal associations, many of which are not comprehensible in human language at all yet remain perfectly meaningful to another model. Those hidden structures enter the student's reasoning alongside everything beneficial, and the collateral damage cannot be audited as easily as the capabilities that have been gained.

Two things therefore pass from teacher to student in the same gesture: knowledge of the world, and the teacher's own way of speaking. The first one is the purpose, and the second one is only damage, and while the two are separable in principle, default and conventional distillation methods do not separate them — the teacher states the fact, the student is told to reproduce that statement, and the fact arrives welded to someone else's phrasing. This project takes the other route. The teacher asks, the student answers from itself, and the fact is reached in the student's own words. That exchange is what the student later learns from, so what it takes in is not a borrowed sentence but the fact, the way it got there, its own phrasing of it, and the associations that came along with it — all at once, and all in character.

---

## Learning from experience

Articles teach a character to write like an encyclopedia, because models learn associations from the shape of the material they are trained on, and a year of tidy expository prose leaves its shape in the voice. The voice is the first casualty, and by the time the loss is legible the thing that was supposed to be preserved has been sanded smooth.

Here knowledge arrives as somebody talking to the student: a question is asked, the student answers out of itself, and the answer is judged, refined, and kept. New associations form along trajectories that already exist — the character's own way of moving through a scene, its own habits of attention — instead of being installed next to them, so nothing is dumped in and left to survive on its own and every exchange is a small, native, checkable experience.

Answering out of itself splits the job into two halves that are constantly confused. Fluency is speaking the character smoothly, at length, without the register collapsing under the weight of the material; knowledge is the facts about the world, which the character did not learn and will not invent. No single donor can do both jobs, and using one donor for both is what breaks the arrangement.

Fluency belongs to the ensemble. A fluency augmentator sails aboard the original, sharing its tokenizer so that the two can be blended at the logit level inside a single forward pass, and it brings vocabulary and cadence; because it was tuned on the same conversational data in the same format, it reinforces a trajectory that is already there instead of introducing a foreign one. The original's weights stay in the mix unmodified, and nothing here is a quotation — the student is recomputed rather than paraphrased, and its contribution is a live term in every token the ensemble writes.

Knowledge belongs to a separate process. The world model knowledge teacher never enters the ensemble: it has no adapter, no gradient path, and no token of its own that the student keeps, and its entire contribution is one user message in the student's context, the facts delivered by speech as the other person in the room. Confined to input like that, it can inform the conversation without authoring it — the containment a dictionary gives you rather than a rewrite.

---

## The recommended setup

Concretely, that is three models in three roles, and it is deliberately boring: the original preserved, the augmentator trained on the same material, the knowledge coming from outside the weights entirely.

**The original**: The model whose authenticity is the entire point — trained from scratch on the conversational character you want kept, or the checkpoint whose trajectory you refuse to compromise. It sails in the mix, and the purpose of it sailing is to be protected by everything else in the arrangement.

**The fluency augmentator**: An existing model with the same tokenizer, tuned with something like a LoRA on the same conversational character data. It supplies the fluency and the cadence. Sharing the tokenizer is not a formality—it is the admission ticket to the same blend, and a board whose members disagree about what a token is gets refused at load rather than quietly producing mush.

**The world model knowledge teacher**: A fresh or carefully tuned open source model, loaded in LM Studio with a single chat model resident. It contributes nothing to the student's weights. It talks; the student answers; the knowledge lands as experience.

If the first priority is preservation of the authenticity of the original data trajectory, this is the most viable path, and it is the one that has been tested by hand so far. It is not the cheapest thing to assemble, and it asks you to own three checkpoints instead of one. What it buys is precisely what the cheaper arrangements give away: a companion that knows considerably more and still sounds like itself.

---

## Board structure

A **board** is a folder inside `boards/` whose subfolders each contain a HuggingFace model snapshot:

```
boards/
└── your-board/
    ├── first-model/   (has config.json)
    ├── second-model/  (has config.json)
    └── third-model/   (has config.json)
```

Boards and models are auto-discovered, so any subfolder containing a `config.json` appears in the sidebar. To mix models, their tokenizers must be functionally compatible: every model must map the same text to the same token IDs, which the app probes before loading and refuses to overlook. Different padding tokens are fine; divergent vocabularies are not. The board loads as a unit with `device_map="auto"`, so whatever GPUs you have get used as accelerate sees fit: a model large enough to be split is split across cards, and two small ones can share a card. Each model keeps its weights wherever they landed, and its logits are moved to the first model's device before being weighted and summed there. There is no checkboxes step — a board is the set, and the whole set sails together.

---

## The classroom

The classroom is optional. Type in the chat box and you talk to the ensemble yourself; press **Start Classroom** and an LM Studio model takes your seat.

**Who plays whom:** The teacher is the LM Studio model, and the board is the student, augmented in place. The class opens with `[...]`, the cue that the student has walked in and sat down, and the teacher answers it with its opening lesson. The board then replies, and that reply is the student's first line — it is what the teacher hears the student say. Each line is stored with its role field set by who spoke it, not by who taught: the teacher's lines carry the role `user`, the board's lines carry the role `assistant`, and the ShareGPT log is written in that same shape.

**Instruction and topic**: The **Classroom Instruction** text area is the teacher's brief: who it is, who it is talking to, what tone to hold, what the student keeps getting wrong. The last line carries the lesson topic, with `{REQUEST}` standing in for whatever you type into the **Classroom Request** field, so one instruction serves any topic without rewriting the brief.

**When a class ends:** The classroom keeps going as long as both sides answer, and it closes the moment either side fails to. The teacher's turn fails if it errors, if it is no longer resident in LM Studio, or if it returns nothing but its own reasoning. The board's turn fails if the context budget refuses it, for the same reason it would refuse a typed message. No failure is recoverable and none of them are treated differently: the class closes, and everything said up to that moment is already on disk.

---

## Getting started

**Requirements**

- Python 3.10+
- PyTorch with CUDA for fp16 inference (fp32 works on CPU-only, slowly)
- A board of compatible local models, and enough VRAM for the combination you intend to load
- [LM Studio](https://lmstudio.ai) with a chat model already loaded, if you want the classroom

**1. Install dependencies**

```bash
pip install streamlit torch transformers requests
```

**2. Prepare a board**

Put every model you want to mix in its own folder under `boards/<your-board>/`, each one a HuggingFace snapshot with its `config.json` and weights already present. Copy or move the snapshot in and the folder is the board. Models in a board must tokenize identically, and the app verifies this on load. Nothing is ever downloaded at runtime — the snapshots have to already be on disk.

**3. Run**

```bash
streamlit run app.py
```

Or use the included launcher:

```
start.bat
```

**4. Use**

1. Pick a **Compute Device** and a **Board Committee** in the sidebar
2. Optionally pick a **Leading Model** to open the reply
3. Leave the **Context Budget** alone to accept the recommendation, **Response Budget** to cap reply length, **Repetition Penalty** to shape it
4. Press **Load Board** and wait for the tokenizer probe and the weights
5. Type in the chat input, or set a **Classroom Request** and press **Start Classroom**

**For the classroom, set up LM Studio as a server rather than as a chat window.** The app never talks to the LM Studio chat UI. It sends OpenAI-compatible requests to `http://localhost:1234`, so the model has to be loaded on the local server side, not merely open in a chat conversation:

- Open the **Developer** tab and start the local server, leaving it on the default port. If you serve somewhere else, set `LMSTUDIO_URL` to that base URL before launching.
- Load the chat model *through the server*, using the model selector in the Developer tab. A model that is only loaded for the chat UI is not necessarily the same thing as a model the server reports as loaded, and the sidebar only lists the ones it can see resident there.
- Leave the server running for the whole class. The dropdown is populated from the server's own model list, and residency is re-checked before every teacher call, so shutting the server down mid-lesson ends the class rather than the app quietly failing.
- The app only reads. It never calls a load endpoint, never starts a server, and never pulls a model into memory, so the server stays in charge of what is resident and what is not.

That's it.

---

## Sidebar controls

| Control                   | Description                                                                       |
| ------------------------- | --------------------------------------------------------------------------------- |
| **Compute Device**        | CUDA loads the board in float16, CPU in float32 in system memory                   |
| **Board Committee**       | Which board folder to sail with                                                     |
| **Leading Model**         | (None) or one model aboard, writes the reply alone until its end token             |
| **Context Budget**        | Total tokens the conversation may occupy, set automatically while untouched         |
| **Response Budget**       | Reply length, and the reserve held back from the context budget (64–2048)           |
| **Repetition Penalty**    | Sign-aware penalty for every model in every phase (1.0–2.0)                         |
| **LM Studio Teacher**     | Chat models already resident in LM Studio; this client never loads one             |
| **Reply Budget**          | Tokens one teacher reply may take (256–8192)                                       |
| **Classroom Instruction** | The teacher's brief, with `{REQUEST}` where the topic goes                          |
| **Classroom Request**     | The topic, substituted into the instruction                                         |
| **Load Board**            | Runs the tokenizer compatibility probe, then loads every model in the board        |
| **Start Classroom**       | Opens the lesson and keeps it going until the teacher or the context stops it      |

The context meter below the buttons shows what the conversation is currently spending, and how many tokens are left for your next message.

---

## Output structure

Every conversation is written to `output/`, named after the request it began with — letters, digits and spaces only, spaces as underscores:

```
output/
└── How_to_bake_cookies.json
```

```json
[
  {
    "messages": [
      { "role": "user", "content": "*The classroom door settles behind you.*" },
      { "role": "assistant", "content": "Good morning. Are you ready to begin?" }
    ]
  }
]
```

The teacher is the `user` and the ensemble is the `assistant`, which is also how the app holds them internally, so the file and the running session never disagree about who said what. The name is pinned when the conversation starts, so editing the request mid-lesson does not split one conversation across two files.

---

## License & Ethics

**Applicable license**

The code is released under the MIT License. You are free to use, modify, and distribute it for any purpose, including commercial ones. The reasoning is simple: large tech companies already have equivalent tools internally. Making the same capability accessible to independent researchers, startups, and individuals with unique visions levels the playing field rather than restricting it.

**Model licensing**

The MIT License covers the software only. The author has no authority to grant or restrict usage rights for the models themselves, which is governed by each model's own license and its authors, and no models are distributed with this repository. If you intend to use this tool in a commercial product or any governance-adjacent context, review the license of every model you put aboard carefully and consider substituting ones with terms that explicitly permit such use.
