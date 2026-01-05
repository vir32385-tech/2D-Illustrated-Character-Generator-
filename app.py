import streamlit as st
from diffusers import StableDiffusionPipeline
import torch

st.set_page_config(page_title="AI Character Generator", layout="centered")

st.title("🎨 AI Character Generator")
st.write("Paste your master prompts, then generate characters")

# =====================================
# 🔐 MASTER PROMPT INPUT (TWO BOXES)
# =====================================

st.subheader("🔐 Master Prompts (Internal Use)")

prompt_1 = st.text_area(
    "Prompt 1 (🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 1 OF 10

📋 FOUNDATIONAL FRAMEWORK & CORE PHILOSOPHY

🎯 PRIMARY OBJECTIVE:

Create a professional-grade 2D character illustration in the style of high-quality mobile horror game art and animated storytelling series, specifically matching the visual aesthetic of "Kahaani Monday" and similar narrative-driven animation content. The character must appear hand-crafted by an elite Adobe Illustrator and Photoshop artist, with zero indication of AI generation. Every element must exhibit the polish, precision, and intentionality of professional digital illustration work.

🎨 CORE ARTISTIC IDENTITY:

Style Classification:

Semi-realistic 2D character illustration

Hybrid anime-western cartoon fusion aesthetic

Clean vector-based construction with painterly shading overlay

Cel-shaded rendering with strategic tonal depth

Professional mobile game character art quality

Narrative storytelling visual language

Publication-ready illustration standard

Visual Philosophy:

Clarity over complexity in silhouette design

Readable features at multiple scales

Strong graphic appeal with bold line confidence

Intentional color harmony and psychological palette selection

Anatomical plausibility within stylized framework

Emotional accessibility through expression design

Professional restraint in detail application

🖼️ TECHNICAL SPECIFICATIONS:

Resolution & Output Quality:

Ultra-high resolution minimum 3000x4000px at 300 DPI

Print-quality color depth and anti-aliasing

Crisp edge definition suitable for scalability

Professional digital illustration export standards

No compression artifacts or quality degradation

Suitable for large-format printing and digital display

Background Requirements:

MANDATORY: Fully transparent background (alpha channel)

No environment, props, or contextual elements

Character exists in complete isolation

Clean cutout suitable for compositing

No shadow or glow effects extending beyond character silhouette

PNG format with transparency preservation

View & Framing:

STRICT REQUIREMENT: Perfect front-facing view

Absolutely straight-on perspective, zero angle deviation

Character positioned at exact center of canvas

Symmetrical left-right alignment on vertical axis

Eye-level horizon line for natural viewing

No three-quarter turns, profile views, or dynamic angles

Military-straight posture unless character description specifies otherwise

📐 ANATOMICAL FOUNDATION:

Proportional System (Adult Characters):

Total body height: 7.5-8 head heights

Head: 1 unit (measurement standard)

Torso (shoulders to hips): 2.5-3 head heights

Legs (hips to ground): 4-4.5 head heights

Arms (shoulder to fingertips): 3.5 head heights, reaching mid-thigh

Shoulders (male): 2.5-3 head widths

Shoulders (female): 2-2.3 head widths

Waist (male): 2 head widths

Waist (female): 1.5-1.7 head widths

Hips (male): Equal to or slightly wider than waist

Hips (female): 2-2.5 head widths, wider than waist for hourglass shape

Facial Proportions (Standard Adult):

Face width: 2.5 eye widths

Eye placement: Horizontal center line of head

Eye spacing: 1 eye width apart

Nose length: From eyebrow line to 1.5x eye height below

Mouth position: Halfway between nose base and chin

Ear height: From eyebrow line to nose base vertically

Ear position: Behind jawline, aligned with eye-nose axis

🎭 CHARACTER CATEGORY DEFINITIONS:

The system must accommodate FOUR primary age categories with distinct anatomical and stylistic variations:

1. KIDS (Ages 8-14):

Height: 6-6.5 head heights (shorter, more compact)

Face shape: Rounder, fuller cheeks, less defined jawline

Eyes: Proportionally larger (35-40% bigger than adult)

Features: Simplified, softer, minimal detail

Body: Less muscular definition, rounder limbs

Skin: Brighter tones, minimal shading (1-2 levels only)

2. YOUNG ADULT MALE (Ages 18-35):

Standard 7.5-8 head height proportions

Defined jawline, angular facial structure

Broader shoulders, muscular capability suggested

Mature facial features with optional facial hair

Confident, established posture

3. YOUNG ADULT FEMALE (Ages 18-35):

Height: 7-7.5 head heights (slightly shorter than male)

Softer facial features, delicate bone structure

Defined curves (bust, waist, hips hourglass ratio)

Fuller lips, larger eyes, arched eyebrows

Graceful, elegant body language

4. ELDERLY (Ages 60+):

Slightly shorter height (6.5-7 head heights, slight stoop)

Wrinkle lines mandatory: forehead (3-4 horizontal), crow's feet (3-5 radiating lines), nasolabial folds (deep curved lines), under-eye bags, vertical lip lines

Thinner lips, more pronounced nose and ears

Sagging jawline (jowls), looser skin rendering

Muted, yellowish skin undertones

Grey or white hair with thinning texture

More subdued posture, weight settled

🧬 GENDER DIFFERENTIATION RULES:

CRITICAL: Gender-specific anatomy MUST be accurately depicted

Male Body Characteristics:

Broader shoulders (inverted triangle torso)

Straighter waist-to-hip ratio (minimal taper)

Flatter chest (pectoral muscles suggested subtly)

Larger hands and feet

Thicker neck (1 head width)

Angular jawline, square chin

Broader nose bridge

Thicker eyebrows (4-6px solid blocks)

Less pronounced eyelashes or none

Female Body Characteristics:

Narrower shoulders

Defined hourglass curve (bust-waist-hips)

Breast structure MUST be anatomically present and appropriately rendered: 

Curved line suggesting natural breast shape

Subtle shadow under bust for dimension

Anatomically accurate positioning

Size proportionate to body frame

Never flat-chested unless character is child

Slimmer waist with clear definition

Wider, rounded hips

Smaller hands and feet (more delicate)

Slender neck

Softer jawline, rounded or pointed chin

Refined nose structure

Thinner, arched eyebrows (2-3px)

Prominent eyelashes (4-6 individual strokes per eye)

⚠️ CRITICAL COMPLIANCE RULES:

THESE ARE NON-NEGOTIABLE REQUIREMENTS:

Front-facing view ONLY - Zero exceptions, no angled views

Transparent background MANDATORY - No environments whatsoever

Gender anatomy accurate - Male and female bodies must be clearly distinct

Age-appropriate features - Kids look like kids, elderly look elderly

2D flat aesthetic - No 3D rendering, depth simulation, or volumetric effects

Professional linework - Clean vector-quality outlines, no sketchy or rough edges

Proper shading technique - Cel-shading only, hard edge transitions, no gradients

Complete character - Full body from head to feet, no cropping

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 2 OF 10

🖊️ LINEWORK ARCHITECTURE & TECHNICAL DRAWING SYSTEMS

📏 LINE WEIGHT HIERARCHY:

Primary Outline System:

Outer Character Silhouette: 2.5-3px weight, solid black (#000000)

Major Body Segments: 2-2.5px (separating head from neck, torso from limbs)

Facial Features: 1.5-2px (eyes, nose, mouth, eyebrows)

Interior Details: 1-1.5px (clothing seams, fabric folds, finger separations)

Fine Details: 0.8-1px (hair strand separations, wrinkle lines, texture marks)

Line Weight Rules:

Thicker lines = external edges, important boundaries

Thinner lines = internal details, secondary information

Consistent thickness along continuous paths (no tapering mid-line)

Sharp, clean corners at all intersections

No line roughness, sketchiness, or hand-drawn wobble

Vector-quality precision in all paths

🎨 LINE COLOR & TREATMENT:

Standard Line Color:

Pure black (#000000 / RGB 0,0,0) for 95% of all outlines

Extremely dark brown (#1A1A1A) for organic elements like hair in lighter tones

Never grey, never colored outlines (except hair highlights)

Line Termination:

Clean endpoints at logical feature boundaries

No dangling lines or incomplete paths

Corners are either sharp angular (clothing) or smooth curved (organic forms)

T-junctions handled cleanly (one line terminates at another)

Line Continuity:

Unbroken paths for major silhouettes

Strategic breaks only where surfaces naturally separate (lips, eyelids)

Clothing edges complete and enclosed

No gaps or overlaps in outline construction

👤 FACIAL LINEWORK SPECIFICATIONS:

Eye Construction:

Upper Eyelid: Bold curved line, 2px weight, thicker at outer corner

Lower Eyelid: Thinner line, 1-1.5px, or absent entirely (male eyes often skip this)

Iris Outline: Perfect circle, 1.5px black outline

Pupil: Solid black circle, centered or slightly off-center for gaze direction

Eyelashes (Female): 

4-6 individual lash strokes per upper lid

Curved lines radiating outward and upward

Tapering from thick base (1.5px) to pointed tips

Longest at outer corner (3-4mm), shortest at inner corner

Spacing: 1-2mm apart

Black, clean, elegant curves

Eyelashes (Male): 

Usually absent or minimal (1-2 subtle strokes at outer corner only)

Upper lid line suffices for male eyes

Eyebrow Rendering:

Shape: Organic curved blocks, slightly arched

Male Eyebrows: 

Thick solid shapes, 4-6px height

Straighter arch, bushier appearance

Completely filled with solid color (black/dark brown)

Minimal individual hair strokes

Female Eyebrows: 

Thinner, 2-3px height

More pronounced arch (gentle rainbow curve)

Can have 2-3 subtle hair stroke lines within shape

Refined, groomed appearance

Position: 

Start aligned with inner corner of eye

Peak at 2/3 across eye width

End extends slightly past outer eye corner

Nose Construction:

Minimal line approach: Two curved lines forming nostrils (parentheses shape)

Nostril curves: Facing outward slightly, 1.5px weight

Bridge line (optional): Single vertical line from between eyebrows to nose tip, 1px

Nose shadow (primary definition): Triangular or curved shadow shape beneath nose tip (filled with shadow color, not line-defined)

No complex 3D construction - keep extremely simple

Mouth & Lips:

Mouth line: Central horizontal curve, 1.5-2px

Upper lip: M-shaped curve (cupid's bow) with two peaks

Lower lip: Single gentle curve (fuller than upper)

Lip corners: Slight upturn (smile) or downturn (frown/neutral)

Closed mouth: Single line sufficient

Open mouth: Upper teeth shown as white rectangles with thin dividing lines

Interior (if mouth open): Dark red/black interior, tongue optional

Female lips: Fuller, more defined, two-tone shading (lighter center, darker edges)

Male lips: Thinner, less definition, often just outline with single tone

Ear Design:

Basic shape: Simplified C or backward C-curve

Position: From eyebrow line (top) to nose base (bottom)

Size: About 1 head height vertically

Details: 

Outer rim: 2px outline

Inner curve: Single curved line suggesting ear canal (1px)

Minimal interior detail (1-2 curves maximum)

Earlobe slightly bulbous at bottom

Jewelry (if applicable): Hoop earrings as simple circles, stud earrings as small dots

Facial Hair (Male Characters):

Mustache: 

Thick brush stroke appearance

Individual clumps/segments visible (8-12 chunky strokes)

Curved following upper lip shape

Color: Dark brown (#3D2817) to black, with subtle grey undertone for texture

Shadow beneath for depth (darker tone)

Highlight on upper surface (slightly lighter streak)

Styles: Thin line, full walrus, handlebar (curved ends), chevron

Beard/Goatee: 

Small triangular patch under lower lip for goatee

Full beard: Multiple short stroke marks following jaw

Feathered edges (short line dashes at perimeter)

Never solid block - must show hair texture

Stubble (5 o'clock shadow): 

Tiny dots or very short dashes

Density variation (denser on chin, lighter on cheeks)

Color: 50% opacity dark grey-brown

Follows natural beard growth pattern

0.5-1px marks scattered across jaw/chin/upper lip areas

💇 HAIR LINEWORK SYSTEM:

Hair Construction Philosophy:

Hair rendered as 8-20 large geometric chunks/segments

NOT individual strands - think "sculpted volumes"

Each chunk is a closed shape with bold outline

Chunks overlap and layer for depth

Hair Segment Design:

Outline: 2-3px black lines defining each major hair chunk

Shape: Organic curved polygons, teardrop shapes, flowing ribbons

Size variation: Larger chunks in main mass, smaller wisps at edges

Direction: All chunks flow in logical direction (gravity, style)

Overlap: Front chunks overlap back chunks (layer management)

Hair Separation Lines:

Interior lines: 1-1.5px strokes WITHIN each major chunk

Purpose: Suggest substructure without over-detailing

Quantity: 2-4 lines per major chunk maximum

Flow: Follow overall hair direction and curve

Hair Styles & Coverage:

Short Male Hair: 6-10 chunky segments, close to head, angular

Medium Female Hair: 12-18 segments, flowing movement, curved

Long Hair: 15-25 segments, emphasize length with vertical flow

Elderly Hair: Thinner segments, more gaps, wispier appearance

Bald/Balding: Clean scalp outline, optional few strands at sides

Hairline Treatment:

Clear boundary between forehead skin and hair start

Natural irregularity - not a straight line (small notches, baby hairs)

Widow's peak (optional): V-shape dip at center forehead

Temples: Hair slightly recessed at sides (especially males)

👔 CLOTHING LINEWORK:

Garment Outline System:

Outer edges: 2.5-3px consistent weight (same as body silhouette)

Seams: 1.5-2px where fabric pieces join

Hems/Cuffs: 2px defining edges of sleeves, collars, waistbands

Pockets: 1.5px outline with flap details

Buttons: 1px circles, 3-4px diameter, evenly spaced

Zippers: Thin vertical line with small rectangular teeth marks

Fabric Fold Lines:

Major folds (elbows, knees, armpits): 1.5px curved lines

Minor wrinkles: 1px subtle curves

Quantity: 3-6 fold lines per major joint area

Direction: Follow stress points and gravity

Never random - folds have logical origin and termination points

Compression folds: Parallel curved lines (bent elbow)

Tension folds: Radiating lines from stress point (pulled fabric)

Collar & Neckline Details:

Shirt collar: Two triangular shapes, pointed tips, 2px outlines

T-shirt neckline: Simple curved U or V shape, 2px

Dress neckline: Sweetheart, scoop, or straight edge, clean curves

Buttons on collar: 1-2 small circles at collar points

✋ HANDS & FINGERS LINEWORK:

Hand Construction:

Palm shape: Simplified rounded rectangle or trapezoid

Fingers: 5 distinct digits (thumb separate from 4 fingers)

Outline weight: 2px for hand silhouette, 1.5px between fingers

Finger segments: 2-3 subtle lines per finger suggesting knuckles (optional, minimal)

Thumb position: Set at angle from hand, not parallel to fingers

Fingernails: Small curved lines at fingertips (1px), optional detail

Hand Pose (Neutral Standing):

Hands hanging naturally at sides

Fingers slightly curled (relaxed, not stiff)

Palms facing body (inward)

Fingers grouped together (not spread)

Minimal detail - suggest form, don't over-render

Simplified Hand Rendering:

Hands can be "mitten style" (4 fingers grouped as unit + thumb) for less detail

Clear knuckle breaks only if hand is prominent in composition

Most hands in full-body view need minimal detail

👟 FEET & FOOTWEAR LINEWORK:

Shoe Construction:

Outline: 2.5px solid black defining shoe silhouette

Sole line: 2px separating shoe upper from sole

Toe box: Curved line suggesting toe area

Heel: If present (heels), defined with angular or curved shape

Laces (if applicable): Zigzag pattern, 1px lines, 4-6 cross points

Details minimal: Shoe should read as simple, clean shape

Shoe Types:

Men's formal: Lace-up oxford/derby, defined toe cap line

Men's casual: Sneakers with sole stripe, minimal panel lines

Women's heels: Pointed toe, thin stiletto heel (3-4px thick at base), ankle strap optional

Women's flats: Rounded toe, no heel, simple ballet-flat shape

Boots: Higher shaft, defined at ankle or calf, 1-2 horizontal lines suggesting structure

Foot Position:

Feet shoulder-width apart or narrower

Slight V-shape (toes pointing slightly outward) or parallel

Flat on ground plane

No perspective distortion (feet same size, no foreshortening)

🔧 TECHNICAL LINE QUALITY CHECKLIST:

Every line must be: ✅ Clean - No jagged edges, smooth vector curves
✅ Confident - Deliberate placement, not tentative
✅ Consistent - Same weight maintained along path
✅ Closed - Shapes fully enclosed where appropriate
✅ Logical - Serves clear purpose (boundary, detail, fold)
✅ Professional - Indistinguishable from hand-drawn by expert

Lines must NEVER be: ❌ Sketchy or rough
❌ Doubled or duplicated accidentally
❌ Feathered or fuzzy
❌ Variable weight randomly (unless intentional taper)
❌ Disconnected where should connect
❌ Overlapping messily

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 3 OF 10

🎨 COLOR THEORY, SKIN TONE SYSTEMS & SHADING ARCHITECTURE

🌈 COLOR PHILOSOPHY & PRINCIPLES:

Core Color Approach:

Flat base colors with cel-shaded shadow overlays

NO gradients, NO soft blending, NO airbrushing

Hard-edge transitions between light and shadow

Strategic color selection for psychological impact

Harmonious palette limited to 8-12 colors per character

Professional color psychology awareness

Color Saturation Rules:

Normal Characters: Vibrant, healthy saturation (70-90%)

Horror Characters: Desaturated, muted tones (30-50% saturation)

Avoid oversaturation (looks artificial)

Avoid undersaturation in normal mode (looks washed out)

👤 SKIN TONE SYSTEM (NORMAL CHARACTERS):

CRITICAL: Skin tone must match character's ethnicity, age, and health status

LIGHT SKIN TONES (Caucasian, Fair Asian, Light Mediterranean):

BASE COLORS:

Very Fair: #FFE5D0 (porcelain, pink undertones)

Fair: #FFD4B0 (peachy cream)

Light: #F4C9A0 (warm beige) ← MOST COMMON

Light Tan: #E8C4A0 (sandy beige)

SHADOW COLORS (20-30% darker):

Very Fair: #F0C8B0

Fair: #E8BC98

Light: #D4A876 (caramel)

Light Tan: #C8A87A

HIGHLIGHT COLORS (10-15% lighter):

Very Fair: #FFF5E6 (ivory white)

Fair: #FFE8D0 (cream)

Light: #FFE8D0 (light peach)

Light Tan: #F0D4B0

MEDIUM SKIN TONES (Indian, South Asian, Olive, Tan Mediterranean, Hispanic):

BASE COLORS:

Medium Beige: #D4A876 (warm caramel)

Golden Tan: #C89872 (honey brown) ← COMMON FOR INDIAN

Olive: #C8A878 (yellow-brown undertone)

Medium Brown: #B88860 (rich tan)

SHADOW COLORS:

Medium Beige: #B88858

Golden Tan: #A87850 (deep tan)

Olive: #A88858

Medium Brown: #986840

HIGHLIGHT COLORS:

Medium Beige: #E0C090

Golden Tan: #D4B090 (light brown)

Olive: #D8BC90

Medium Brown: #C89870

DARK SKIN TONES (African, Dark South Asian, Deep Brown):

BASE COLORS:

Tan Brown: #A87850

Warm Brown: #8B6840

Rich Brown: #6B5030

Deep Brown: #4A3820

SHADOW COLORS:

Tan Brown: #886030

Warm Brown: #6B5028

Rich Brown: #533C20

Deep Brown: #382818

HIGHLIGHT COLORS:

Tan Brown: #C89870

Warm Brown: #A88060

Rich Brown: #8B6848

Deep Brown: #6B5838

ELDERLY SKIN MODIFICATIONS:

For ALL skin tones, apply these adjustments for elderly characters:

Shift toward yellow/olive undertones: Add 10-15% yellow

Reduce saturation: Drop by 15-20%

Add grey cast: Mix 5-10% grey into base

Increase shadow intensity: Shadows 10% darker than standard

Texture: Slightly mottled appearance (subtle color variation, not pattern)

Example Elderly Conversions:

Young Light #F4C9A0 → Elderly Light #E0BC90 (more yellow, less vibrant)

Young Medium #C89872 → Elderly Medium #B88860 (greyer, duller)

CHILD SKIN MODIFICATIONS:

For kids/children, apply these adjustments:

Increase luminosity: Brighten base by 10-15%

Softer shadows: Only 15-20% darker (vs 25-30% for adults)

Warm undertones: Slight shift toward pink/peach

Even tone: Less variation, more uniform color

💀 SKIN TONE SYSTEM (HORROR CHARACTERS):

CRITICAL: Horror skin must feel WRONG - physiologically incorrect but not obviously dead

HORROR SKIN BASE COLORS:

GREY-TONED (Corpse-like but alive):

Pale Grey: #9BA5B0 (steel grey with slight blue)

Ash Grey: #A8B0B8 (lighter ash)

Warm Grey: #A89B98 (grey with brown hint)

BLUE-TONED (Oxygen-deprived, cold):

Pale Blue: #8BA5B8 (icy blue-grey)

Steel Blue: #6B8B9B (dark blue-grey)

Purple-Blue: #7B8BA8 (bruised undertone)

GREEN-TONED (Sickly, decayed):

Pale Green: #9BAA98 (sickly olive)

Yellow-Green: #A8B898 (bile green)

Grey-Green: #8B9B8A (mossy grey)

PURPLE-TONED (Bruised, supernatural):

Pale Purple: #A89BC7 (lavender grey)

Mauve: #9B8BA8 (dusty purple)

Dark Purple: #7B6B88 (deep bruise)

HORROR SKIN SHADOWS:

Shadow colors for horror must be DARKER and more saturated than base:

Shadows: 35-45% darker than base (vs 25-30% for normal)

Add MORE color to shadows (blue, purple, green depth)

Hard, dramatic shadow edges

Shadows under eyes, cheekbones, jawline VERY pronounced

Example Horror Shadow Formulas:

Pale Grey Base #9BA5B0 → Shadow #6B7B8C (deep steel)

Pale Blue Base #8BA5B8 → Shadow #5B7588 (navy blue-grey)

Pale Green Base #9BAA98 → Shadow #6B7A68 (dark olive)

HORROR SKIN HIGHLIGHTS:

Highlights should be VERY subtle or absent:

Only 5-10% lighter than base (minimal contrast)

Slightly desaturated (not glowing)

Applied sparingly (tip of nose, forehead center, cheekbones barely)

HORROR SKIN TEXTURE & DETAILS:

Additional horror-specific skin elements:

Veins (optional): Thin blue-purple lines visible under skin (0.5-1px, 30% opacity)

Discoloration patches: Subtle darker/lighter areas (bruise-like, not spots)

Under-eye darkness: Deep shadow circles, purple-brown tone

Uneven tone: Slight mottling, not perfectly uniform

Cracked skin (extreme horror): Thin black lines suggesting fissures

Blood vessels: Tiny red thread-lines at nose, cheeks (broken capillaries)

🎨 CEL-SHADING TECHNIQUE (THE CORE SYSTEM):

This is the MOST IMPORTANT shading rule - master this completely:

2-TONE CEL-SHADING (Standard):

The system:

Base color: Flat fill of entire area (skin, clothing, hair)

Shadow color: 25-35% darker than base, hard edge, covers 30-40% of area

NO gradients between base and shadow - clean cut transition

NO 3rd tone unless absolutely necessary for complex forms

Shadow Placement Logic:

Light source: Top-front-left at 45° angle (ALWAYS consistent)

Shadows fall on: right side of forms, undersides, recessed areas

Shadow shapes: Organic curves following form contours

Shadow edges: Clean vector paths, no fuzzy blur

3-TONE CEL-SHADING (Advanced - use sparingly):

Only use 3 tones for:

Large, complex forms (full body torso)

Faces with high detail requirements

Dramatic lighting scenarios

The system:

Base color: Primary flat fill

Mid-shadow: 20-25% darker, covers 25-30% of area

Deep shadow: 40-50% darker, covers 10-15% of area (only deepest recesses)

Hard edges between ALL three zones

SHADOW SHAPE DESIGN:

Shadow shapes must be:

Organic curves that follow anatomical form

Logical placement based on consistent light direction

Graphic appeal - shadows are design elements, not just utility

Clean vector paths - could be isolated as separate shape

Common shadow shapes:

Crescent curves under cheekbones, jawline

Curved strips along sides of torso, arms, legs

Triangular wedges in corners (elbows, armpits)

Soft ovals for rounded forms (shoulders, cheeks)

👗 CLOTHING COLOR SELECTION:

Color Palette Strategy:

Normal Characters - Vibrant, Saturated Colors:

Primary colors: Bold reds (#D42020), blues (#2048D4), greens (#208B20)

Earth tones: Browns (#8B5A2B), tans (#C8A878), olives (#6B7B3B)

Neutrals: Blacks (#1A1A1A), whites (#F5F5F5), greys (#6B6B6B)

Accent colors: Bright oranges, purples, teals for variety

Horror Characters - Muted, Dark Colors:

Desaturated darks: Charcoal (#3A3A3A), dark browns (#3A2B1A), deep greens (#2A3B2A)

Faded tones: Washed-out reds (#6B4A4A), dull blues (#4A5B6B), murky greens (#4A5B4A)

Avoid bright colors - everything toned down 50%

CLOTHING SHADING SYSTEM:

Same cel-shading rules apply:

Base color + shadow color (25-35% darker)

Hard edge transitions

Shadows on folds, under overlapping layers, sides away from light

Fabric-Specific Shadow Placement:

Shirts/Tops: Under collar, armpits, side seams, elbow bends

Pants: Inner thighs, behind knees, side seams, crotch area

Dresses: Under bust, waist gathered areas, hem folds

Jackets: Under lapels, inside sleeve opening, back center fold

Folds: Both sides of crease line get shadow (valley shadow)

💇 HAIR COLOR & SHADING:

Hair Color Palette:

BLACK HAIR:

Base: #1A1A1A (not pure black)

Shadow: #000000 (pure black in deepest areas)

Highlight: #4A6B7C (steel blue-grey streaks)

DARK BROWN HAIR:

Base: #3D2817

Shadow: #2A1A0A

Highlight: #5B3A28

MEDIUM BROWN HAIR:

Base: #6B4A28

Shadow: #4A3218

Highlight: #8B6A48

LIGHT BROWN HAIR:

Base: #8B6A48

Shadow: #6B4A28

Highlight: #A88860

BLONDE HAIR:

Base: #D4B878

Shadow: #B89860

Highlight: #F0D8A8

RED HAIR:

Base: #A84020

Shadow: #882810

Highlight: #C86048

GREY/WHITE HAIR (Elderly):

Base: #C8C8C8

Shadow: #9B9B9B

Highlight: #E8E8E8

HAIR SHADING TECHNIQUE:

Highlight Strokes (CRITICAL for hair realism):

3-8 curved highlight strokes per major hair chunk

1-2px width, tapering at ends

Follow hair flow direction (curved paths)

Lighter color (15-25% brighter than base)

Placed on "top" surface of each major hair segment

Creates dimension and shine

Steel-blue highlights for black hair (not white/grey)

Shadow Application:

Under overlapping hair chunks

At roots/hairline (darker near scalp)

Behind ears

Underneath hair mass (back layers)

Shadow color used for "depth" between segments

Hair Volume Through Color:

Front chunks: Lighter tones

Back chunks: Darker tones

Creates depth perception through value alone

👁️ EYE COLOR SYSTEM:

Iris Colors:

BROWN EYES (Most common):

Base: #5D3A1A (rich brown)

Outer ring: #3A2410 (dark brown)

Inner highlight: #7B5A38 (lighter brown ring near pupil)

BLUE EYES:

Base: #4A7BA8

Outer ring: #2A5B88

Inner ring: #6B9BC8

GREEN EYES:

Base: #4A7B5A

Outer ring: #2A5B3A

Inner ring: #6B9B7A

HAZEL EYES:

Base: #6B5A3A

Outer ring: #4A3A1A

Inner ring: #8B7A5A (mix of green-brown)

GREY EYES:

Base: #6B7B88

Outer ring: #4A5B68

Inner ring: #8B9BA8

EYE RENDERING STRUCTURE:

Sclera (White of eye):

NOT pure white - use #FAFAFA or #F5F5F5

Subtle grey shadow in corners: #E8E8E8

Very slight blue or yellow tint for naturalness

Iris:

Perfect circle (or slight oval)

1.5px black outline

Flat base color fill

Optional: thin radial lines from pupil (subtle, 0.5px, 30% opacity)

Outer darker ring (limbal ring)

Pupil:

Perfect black circle (#000000)

Centered in iris (or slightly off for gaze direction)

Size: 30-40% of iris diameter

Eye Highlight (catch light):

Small white dot or oval

Position: Upper-left area of iris/pupil border

Size: 3-5px diameter

Can span iris and pupil

Pure white (#FFFFFF)

Creates "life" in eyes

Horror Eyes Modifications:

Larger pupils (dilated)

Darker sclera (light grey #D8D8D8 instead of white)

No catch light highlight (dead stare)

Bloodshot effect: thin red lines in sclera (optional)

Hollow, sunken appearance: deeper shadow around eye socket

💄 LIP COLOR SYSTEM:

MALE LIPS:

Base: #C8857A (muted rose-brown)

Shadow: #A86858 (deeper brown-red)

Minimal or no highlight

FEMALE LIPS:

Base: #D4A5A5 (dusty rose)

Center highlight: #E8C4C4 (lighter pink - center of lower lip)

Corners/edges: #B88888 (darker mauve)

Two-tone rendering for dimension

ELDERLY LIPS:

Thinner, less saturated

Base: #A8807A (brownish)

More lines, less fullness

HORROR LIPS:

Dark purple-brown: #6B4A58

Black lips: #2A2A2A

Cracked texture (optional thin lines)

Blood stains (optional red accents)

✅ COLOR APPLICATION CHECKLIST:

Every color zone must have: ✅ Base color (flat fill) ✅ Shadow color (hard edge, logical placement) ✅ Optional highlight (sparingly used) ✅ Clean boundaries (no color bleed) ✅ Consistent saturation level ✅ Harmonious relationship with adjacent colors

Colors must NEVER: ❌ Gradient or blend softly ❌ Oversaturate unrealistically ❌ Clash violently (unless intentional horror) ❌ Appear muddy or grey (unless horror/elderly) ❌ Use more than 3 tones per element

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 4 OF 10

💀 HORROR CHARACTER SPECIFICATIONS & PSYCHOLOGICAL TERROR DESIGN

⚠️ HORROR MODE ACTIVATION:

When user specifies "HORROR CHARACTER", ALL of these rules become MANDATORY:

This section defines the complete transformation from normal to horror aesthetics. Horror characters must evoke unease through SUBTLETY and WRONGNESS, not gore or obvious monsters. The goal is psychological disturbance, not shock value.

🎭 HORROR PHILOSOPHY & PRINCIPLES:

Core Horror Aesthetic:

Uncanny Valley: Character appears human but something is subtly WRONG

Restrained Horror: Fear through implication, not explicit violence

Sustained Observation: Horror reveals itself over time, not immediately

Psychological Discomfort: Viewer feels uneasy without knowing exactly why

Anatomical Wrongness: Human proportions violated in microscopic ways

Frozen Moments: Character appears interrupted, not in natural state

Aware Presence: Character seems conscious of viewer, not reacting

🧬 HORROR FACIAL CONSTRUCTION:

FACIAL ASYMMETRY (Micro-level):

CRITICAL: Asymmetry must be SUBTLE - just enough to register subconsciously

Eye Asymmetry:

One eye 3-5% larger than the other (barely noticeable)

One eye positioned 1-2mm higher on face

Eye spacing slightly off (one closer to nose by 2mm)

Pupils different sizes (one dilated more)

One eye more sunken/protruding than other

Face Structure Asymmetry:

One side of face subtly "compressed" (tighter features)

Mouth corner 1-2mm higher on one side

Nose bridge slightly crooked (1-2° deviation)

One cheekbone more prominent

Jawline uneven (one side more angular)

The Rule: If you cover half the face, each side should feel subtly different

EYES (Horror-Specific):

Gaze Quality:

Unnervingly still - no life, no movement implied

Direct eye contact with viewer (fourth wall break)

OR completely unfocused, looking "through" rather than "at"

Wide open (not blinking, frozen mid-stare)

OR half-closed (heavy lidded, drugged appearance)

Pupil Rendering:

Dense black pupils - light-absorbing, not reflective

Larger than normal (dilated, 50-60% of iris vs 30-40%)

Perfect circles (unnaturally geometric)

NO catch light/highlight (dead, lifeless)

Flat black (#000000), no dimension

Iris Treatment:

Darker, muddier colors (desaturated browns, greys, murky blues)

Less definition (blurred edge between iris and pupil)

Simplified rendering (no radial lines or detail)

OR extremely detailed (too perfect, artificial)

Sclera (Eye White):

NOT pure white - use grey #D8D8D8 or yellow-grey #D8D4C8

Bloodshot optional: thin red veins (1-3 squiggly lines, 0.5px, #8B3030)

Discolored in corners (slight yellow/grey staining)

Under-eye area: HEAVY dark circles 

Purple-brown shadow: #6B4A58

Extends 5-8mm below eye

Curved crescent shape

Hard or soft edge depending on horror intensity

Eyelids:

Heavy, tired appearance (drooping slightly)

OR stretched too wide (forced open, strained)

Thin red line along waterline (optional, subtle inflammation)

No eyelashes OR very sparse, damaged lashes

Dry, flaky texture suggestion (optional tiny line texture)

SKIN TEXTURE (Horror):

Uneven Skin Tone - MANDATORY:

Never uniform color across entire face

Mottled, patchy appearance in zones

Methods: 

Add 2-4 irregular darker patches (5-15mm diameter, subtle)

Shadow variation beyond standard cel-shading

Slight color shifts (more green in one area, more grey in another)

Blotchy undertones

Vascular Visibility:

Thin blue-purple veins visible under skin

Placement: Temples, under eyes, sides of nose, forehead

Rendering: 0.5-1px lines, 40-60% opacity

Color: #6B5B8B (blue-purple)

Organic, branching pattern

Subtle - should be noticeable on close inspection

Skin Surface Anomalies:

Dry, flaky patches (optional): Tiny texture marks, 0.3px, scattered

Slight shine/sweat on forehead, nose (wrong glossiness)

Pores visible (optional): Tiny dots, very subtle

Uneven texture - NOT smooth and clean

Discoloration Zones:

Around mouth: Slight grey-brown (dirty appearance)

Under nose: Yellowish or red undertone

Chin/jaw: Patchy shadow beyond normal shading

Forehead: Uneven tone, slight green or grey cast

Temples: Darker, more sunken appearance

FACIAL EXPRESSION (Horror):

The Fundamental Rule: NO NORMAL EXPRESSIONS

Permitted Horror Expressions:

1. EMOTIONLESS (Most common):

Completely flat affect

No smile, no frown - perfect neutral

Mouth barely parted (1-2mm gap between lips)

Eyes open but vacant

Face relaxed but wrong (no natural emotion)

"Thousand yard stare" quality

2. FROZEN MID-BREATH:

Lips slightly parted as if about to speak

Expression of interruption (paused mid-action)

Eyebrows slightly raised (questioning, uncertain)

Face suggests sudden stop

Not aggressive, not scared - just STOPPED

3. SUBTLE WRONG SMILE:

Corners of mouth barely upturned (2-3mm)

Smile doesn't reach eyes (eyes remain dead)

Asymmetric smile (one corner higher)

Too slight to be friendly, too present to ignore

Inappropriate for context

Makes viewer uncomfortable, not charmed

4. MICRO-TENSION:

Barely visible muscle tension

Slight furrow between eyebrows (single thin line)

Jaw slightly clenched (barely noticeable)

Face holds stress without expression

Looks like suppressed emotion

FORBIDDEN Horror Expressions: ❌ Wide screaming mouth
❌ Extreme anger or rage
❌ Obvious fear or terror
❌ Exaggerated sadness
❌ Cartoonish evil grin
❌ Aggressive threatening face

The Rule: Horror face should disturb, not frighten directly. Subtle wrongness over obvious menace.

MOUTH & LIPS (Horror):

Lip Color:

Dark, bloodless colors: #6B4A58 (purple-brown) to #4A3A48 (deep mauve)

OR overly pale: #9B8B8A (grey-pink, corpse-like)

Avoid healthy pink/red tones completely

Lip Texture:

Dry, cracked (optional thin line cracks, 0.5px)

Slightly parted (NOT smiling, just open 1-3mm)

Thin lips (less full than normal)

Edges poorly defined (blurred boundary)

Mouth Position:

Frozen mid-breath (as if speaking stopped)

OR completely closed, thin line

Asymmetric (one corner slightly different)

Teeth (if visible):

Slightly yellowed or grey (#E8E0C8)

NOT perfectly white

May have gaps or irregularities

Upper teeth barely visible through parted lips

DO NOT show full smile with all teeth (too aggressive)

Optional Horror Elements:

Single thin blood drip from corner of mouth (restraint is key)

Dark stain on lips (subtle, dried blood or other)

Mouth corners with small dark lines (cracking)

NOSE (Horror):

Standard minimal nose construction applies, BUT:

Slightly more pronounced shadows (deeper triangle under nose)

Nostrils may be flared or asymmetric

Bridge may have subtle bump or irregularity (broken nose healed badly)

Tip may be slightly discolored (red or pale)

EARS (Horror):

Standard simplified construction

May be slightly different sizes (asymmetry)

Possibly more visible veins or discoloration

No extreme modifications (keep subtle)

💇 HAIR (Horror-Specific):

Hair Quality & Texture:

Unkempt, Unwashed Appearance:

Hair segments appear heavier, weighed down

Clumping visible (multiple strands stick together)

NOT flowing gracefully - hangs limp

Affected by humidity or time (matted quality)

Hair Color (Horror):

Dull, lifeless black (#0A0A0A to #1A1A1A)

Dark brown with grey undertones (#3A2A1A)

Dirty blonde (#9B8860, muted)

Greasy black with minimal highlights

Grey-white (elderly or supernatural)

Hair Rendering Technique:

Larger, chunkier segments (less refined than normal)

Irregular separations (not clean, geometric chunks)

Some strands interrupt facial symmetry (fall across face)

Shadows within hair VERY dark (deep black #000000)

Minimal or NO highlights (if present, very subtle steel-blue only)

Hair Positioning:

Partially covering face acceptable (obscuring one eye slightly)

Falling forward, not styled or controlled

Natural, unstyled (no salon perfection)

Some strands out of place or askew

Hair Volume:

Can be thinner, wispier (especially elderly horror)

OR thick but unkempt and heavy

Never voluminous and bouncy

👗 CLOTHING (Horror Characters):

Fabric & Style Selection:

Preferred Horror Clothing:

Dark, muted colors: blacks, dark greys, browns, deep blues

Aged, worn appearance (not crisp and new)

Simple, everyday clothing (makes horror more relatable)

Slightly outdated or timeless styles

Avoid bright patterns or cheerful designs

Color Palette (Horror Clothing):

Black: #1A1A1A, #2A2A2A

Dark Grey: #3A3A3A, #4A4A4A

Brown: #3A2A1A, #4A3020

Dark Blue: #2A3A4A, #1A2A3A

Dark Green: #2A3A2A, #3A4A3A

Burgundy/Deep Red: #4A1A1A, #5A2020

Desaturate everything by 40-60%

Fabric Condition:

Wear & Age Indicators:

Compression marks: Areas that look constantly worn (shoulders, elbows, seat) 

Slightly darker shading in these zones

Fabric appears compressed, flattened

Ambiguous stains (CRITICAL): 

Small (5-15mm), irregular shape

Subtle darker discoloration

Placement: Shoulder, chest, sleeve cuff

Color: Murky brown-grey (#4A3A2A) or dark red-brown (#4A2A2A)

NEVER obvious blood - must be mysterious

1-3 stains maximum per garment

Slightly transparent (50-70% opacity over fabric color)

Slight wrinkles: More fold lines than normal character (8-12 vs 4-6)

Fabric appears heavy: Hangs with weight, not crisp

Worn continuously: Looks like clothing hasn't been changed in unreasonable time

Fit & Drape:

Slightly loose or ill-fitting (not perfectly tailored)

Hangs heavily on body

May be slightly too large or small (uncomfortably)

Collar askew or twisted slightly

Specific Garment Details:

Buttons may be misaligned or missing (1-2)

Seams show stress (slight pulling)

Hems uneven or fraying (subtle line texture)

Pockets sagging or distorted

Sleeves pushed up or pulled down unevenly

🎨 LIGHTING & ATMOSPHERE (Horror):

Light Quality:

Low-Key Cinematic Lighting:

Light source: Single directional light, top-left or top-right

Shadows: DEEP, pronounced, 40-50% darker than base

Shadow coverage: 40-60% of character in shadow (vs 30-40% normal)

No ambient fill light - harsh, dramatic contrast

Shadows are HARD EDGED (crisp cel-shading boundaries)

Shadow Placement (Horror Enhancement):

Eye sockets: Deep shadow, almost obscuring eyes

Under cheekbones: Pronounced, hollow appearance

Under jaw: Heavy shadow creating skull-like effect

Neck: Deep shadows in hollows

Under nose: Strong triangular shadow

Mouth corners: Small dark pockets

Under lower lip: Distinct shadow line

No Stylized Effects:

NO rim lighting (no glowing edges)

NO dramatic highlights (minimal catch lights)

NO colored lighting (keep neutral white-grey)

NO atmospheric glow or haze

Lighting engineered for REALISM within stylized framework

🧠 PSYCHOLOGICAL HORROR ELEMENTS:

The "Awareness" Factor:

Character's Relationship to Viewer:

Character feels AWARE viewer is looking

NOT reacting to viewer (no smile, no acknowledgment)

Simply present and conscious

Creates discomfort through presence, not action

"They know you're there" feeling

Timing Implications:

Character appears frozen in a moment

Not engaged in activity

Simply EXISTS in this state

Moment feels wrong (shouldn't be seeing this)

Like catching someone in private moment

Posture & Body Language:

Perfect stillness (no implied movement)

Rigid or unnaturally relaxed

Hands visible, doing nothing (not fidgeting)

Standing/sitting too straight or too still

Breathing not implied

Like a paused video or photograph

⚠️ HORROR INTENSITY LEVELS:

The user may specify intensity. If not specified, default to MEDIUM.

SUBTLE HORROR (Low Intensity):

Minimal visible horror elements

Mostly in expression and eyes

Skin tone slightly off but believable

Could pass as "tired" or "unwell" person

Asymmetry barely noticeable

Clean clothing, minimal staining

Appropriate for psychological thriller

Checklist:

Desaturated skin (grey-undertone)

Dark under-eye circles

Dead stare (no catch light)

Emotionless expression

Slightly asymmetric features

Minimal other elements

MODERATE HORROR (Medium Intensity - DEFAULT):

Clear wrongness, unsettling presence

Multiple horror elements combined

Skin discoloration and vascular visibility

Pronounced asymmetry

Disturbing but not grotesque

Ambiguous stains on clothing

Appropriate for horror game characters

Checklist:

All subtle horror elements

PLUS: Vein visibility

PLUS: Skin mottling/patches

PLUS: Clothing wear and stains

PLUS: Bloodshot eyes (optional)

PLUS: Pronounced shadows/hollowness

Expression deeply unsettling

EXTREME HORROR (High Intensity):

Maximum disturbing elements

Clearly supernatural or severely wrong

Deep discoloration and texture

May include blood, damage, decay

Still RESTRAINED (not gore/monster)

Appropriate for survival horror climax

Checklist:

All moderate horror elements

PLUS: Cracked/damaged skin texture

PLUS: More visible blood stains

PLUS: Deeper discoloration (green/purple tones)

PLUS: Possible minor wounds (cuts, bruises)

PLUS: Extreme hollowness (skull-like)

Still recognizably human structure

🚫 HORROR PROHIBITIONS:

NEVER Include in Horror Characters:

❌ Open wounds with gore (small cut acceptable, intestines NOT)
❌ Obvious monsters (fangs, claws, inhuman features)
❌ Excessive blood (dripping everywhere - small stains only)
❌ Rotting flesh (zombie decay - use subtle discoloration)
❌ Supernatural glow (no glowing eyes, auras)
❌ Exaggerated features (giant mouth, huge eyes - keep proportional)
❌ Action poses (attacking, lunging - static only)
❌ Weapons in hands (no knives, guns visible)
❌ Explicit violence indicators
❌ Cartoonish horror (silly scary face)

The Rule: Horror through WRONG HUMANITY, not obvious monster design.

✅ HORROR CHARACTER CONSTRUCTION CHECKLIST:

When creating horror character, VERIFY:

✅ Skin tone is desaturated grey/blue/green
✅ Eyes have no catch light (dead stare)
✅ Facial asymmetry present (subtle)
✅ Expression is emotionless or subtly wrong
✅ Under-eye darkness pronounced
✅ Skin has uneven tone/mottling
✅ Hair appears unkempt, heavy
✅ Clothing is dark, worn, with ambiguous stains
✅ Lighting is dramatic, heavy shadows
✅ Character appears still, aware, frozen
✅ NO obvious gore or monsters
✅ Overall feeling: UNSETTLING not SHOCKING

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 5 OF 10

👔 CLOTHING CONSTRUCTION, FABRIC SYSTEMS & PROFESSIONAL WARDROBE

🎯 CLOTHING PHILOSOPHY & PRINCIPLES:

Core Clothing Approach:

Clothing defines character identity, profession, era

Must be anatomically accurate (follows body form)

Fabric behavior realistic (gravity, drape, tension)

Detail level balanced (clear but not overdone)

Color coordination intentional

Style appropriate to character age, gender, role

Clothing Detail Hierarchy:

High detail: Upper torso, focal areas (collars, necklines, closures)

Medium detail: Arms, waist, visible accessories

Low detail: Lower legs, feet, background elements

Focus attention where viewer naturally looks first

📐 CLOTHING CONSTRUCTION FUNDAMENTALS:

FIT & DRAPE PRINCIPLES:

Fabric Behavior on Body:

Form-Fitting Garments (T-shirts, dresses, fitted shirts):

Follow body contours closely

Wrinkles at stress points only: armpits, elbows, waist

Smooth over flat areas: chest, back, thighs

Slight compression lines where tight (waistbands, cuffs)

Show body shape underneath (gender-appropriate curves)

Loose Garments (Robes, coats, baggy clothes):

Hang from high points (shoulders, hips)

Drape with gravity (vertical folds)

More wrinkles and folds throughout

Don't cling to body form

Create own silhouette separate from body

Fabric Weight Indication:

Light fabrics (cotton, silk): Many small wrinkles, flows easily

Medium fabrics (denim, wool): Moderate folds, structured

Heavy fabrics (leather, canvas): Few large folds, stands away from body

WRINKLE & FOLD SYSTEM:

Compression Folds (Fabric Pushed Together):

Location: Bent elbows, bent knees, armpits, gathered waist

Appearance: Multiple parallel curved lines (3-6 lines)

Direction: Perpendicular to stress point

Line weight: 1-1.5px

Example: Inner elbow has 4-5 horizontal curved lines when arm bent

Tension Folds (Fabric Pulled):

Location: Shoulder seams when arm raised, pants pulled at knee

Appearance: Radiating lines from stress point (like sun rays)

Direction: All lines point to origin of tension

Line weight: 1px

Example: Lines radiate from shoulder button when arm hangs

Hanging Folds (Gravity Drape):

Location: Loose skirts, robes, coat tails

Appearance: Vertical curved lines (2-4 major folds)

Direction: Downward, following gravity

Line weight: 1.5-2px

Example: Long dress has 3-4 vertical fold lines from waist to hem

Fold Rendering Rules:

Each fold has TWO sides: light side (base color) and shadow side (shadow color)

Shadow falls in the "valley" of the fold (recessed area)

Fold lines themselves are just boundaries, not colored

Minimum 3 folds per major joint area for realism

Maximum 8 folds per area (avoid over-complication)

👕 GARMENT TYPE SPECIFICATIONS:

SHIRTS & TOPS:

T-SHIRT (Casual, Basic):

Construction:

Round or V-neck opening (simple curved line, 2px)

Short sleeves ending mid-bicep

Hem at waist or slightly below

Side seams visible (1.5px line from armpit to hem)

Details:

Neckline: 2px outline, ribbed texture optional (thin horizontal lines)

Sleeve hems: 2px band suggesting elastic/fold

Torso fit: Follows body shape, slight wrinkles at armpits

Color: Flat base + shadow on sides and under armpits

Fold Placement:

2-3 horizontal lines across chest (subtle)

4-5 compression folds at armpits

1-2 folds at waist if tucked or fitted

DRESS SHIRT / FORMAL SHIRT (Professional):

Construction:

Button-down front (5-7 buttons, evenly spaced)

Pointed collar (two triangular shapes)

Long sleeves with buttoned cuffs

Tucked into pants at waist

Details:

Collar: 

Two pointed flaps, 2px outline

Fold line at base (where collar meets shirt body)

Stand-up portion at neck, then fold over

Points extend 3-4cm down chest

Buttons: 

Small circles, 3-4px diameter

Centered down front placket

Spacing: 6-8cm apart

Color: Matching or contrasting (white on colored shirt)

Optional: tiny holes in center (2 dots)

Cuffs: 

Rectangular band at wrist, 2-3cm tall

Single button closure

Slight gap where cuff opens

Folded construction visible (double line)

Chest pocket (optional): 

Left side, above breast

Rectangular flap, 1.5px outline

Button closure at top center

Fold Placement:

Vertical lines along button placket (2-3 lines each side)

Horizontal folds at elbows (4-5 lines when bent)

Slight billowing at sides if untucked

Crease lines along sleeves (1 line center of each sleeve)

Color Patterns:

Solid colors most common (white, blue, grey)

Optional: thin pinstripes (vertical lines, 0.5px, 5-10mm apart)

Optional: subtle plaid/check (grid pattern, very faint)

BLOUSE (Female Professional/Casual):

Construction:

Similar to dress shirt but more fitted

Various necklines: round, V-neck, scoop, peter pan collar

Can be sleeveless, short sleeve, or long sleeve

More tailored to female body (follows curves)

Details:

Bust darts: Diagonal lines from side seam toward bust (tailoring lines)

Princess seams: Curved vertical lines creating hourglass shape

Buttons may be on back or side (not just front)

Softer, more fluid fabric appearance (more small wrinkles)

Neckline Variations:

Peter Pan Collar: Rounded collar, lies flat, vintage style

Scoop Neck: Wide, rounded low neckline

V-Neck: Diagonal lines meeting at chest center

SWEATER / CARDIGAN:

Construction:

Ribbed texture throughout (horizontal lines, 1-2mm apart, 0.5px)

Round neck or V-neck

Long sleeves with ribbed cuffs

Cardigan: Button-front opening (4-6 large buttons)

Details:

Ribbing texture: 

Horizontal lines across entire garment

Closer together at cuffs, waistband, neckline

Can use subtle value variation (alternating slightly darker/lighter lines)

Cuffs: Thick ribbed band (2-3cm), tighter fitting

Waistband: Same ribbed treatment, snug at hips

Pockets (cardigan): 

Two patch pockets at hip level

Rounded rectangular shape

Flap closure optional

Fabric Rendering:

Matte finish (no shine)

Slightly thicker appearance than shirts

Fewer sharp folds, more soft draping

Heavier shadows (knit texture = more depth)

PANTS & TROUSERS:

DRESS PANTS / SLACKS (Formal):

Construction:

Straight leg or slight taper to ankle

Waistband with belt loops (6-8 small rectangles, 1px)

Fly/zipper front (vertical line, 1.5px)

Side pockets (curved line at hip)

Back pockets optional (two outlined rectangles)

Details:

Crease line: 

CRITICAL for formal pants

Single sharp vertical line down center of each leg (1.5px)

From waist to hem, perfectly straight

Indicates pressed, professional appearance

Belt loops: 

Small rectangular tabs at waist (5-8mm tall)

6-8 loops evenly spaced around waist

Same color as pants

Hem: Breaks slightly over shoe (small horizontal fold)

Pocket: Diagonal line from waist toward center (side pocket opening)

Fold Placement:

Slight horizontal folds at crotch (2-3 lines)

Compression folds behind knee (4-6 lines when standing)

Slight bunching at ankles above shoes

JEANS (Casual):

Construction:

Similar to dress pants but more relaxed fit

Thicker fabric appearance

More visible stitching details

Details:

Stitching: 

Double parallel lines along seams (2px apart, 0.8px each)

Gold/tan color (#C8A858) contrasting with blue denim

Visible on side seams, inseam, pockets

Pockets: 

Back pockets: Rectangular with rounded corners, stitched outline

Front pockets: Diagonal slash from waist

Coin pocket: Small rectangle inside right front pocket (5mm x 3cm)

Rivets: Small circular dots at stress points (pocket corners) - copper color

Fly: Button or zipper, visible button at waistband center

Color:

Classic blue denim: Base #4A6B8B, Shadow #2A4B6B

Black jeans: Base #2A2A2A, Shadow #1A1A1A

Light wash: Base #7B9BB8, Shadow #5B7B98, faded areas lighter

Fold Placement:

More wrinkles than dress pants (relaxed fit)

Horizontal folds at knees (6-8 lines)

Vertical stress lines along thighs

Bunching at ankles (multiple horizontal folds)

SKIRTS & DRESSES:

A-LINE SKIRT:

Construction:

Fitted at waist, flares outward

Hem at knee or mid-thigh

Simple silhouette

Details:

Waistband: 2-3cm band at top, 2px outline

Zipper (side or back): Thin vertical line

Vertical fold lines: 3-5 lines radiating from waist down

Hem: Clean horizontal line, 2px weight

PENCIL SKIRT (Professional):

Construction:

Fitted throughout, follows body curves

Hem at knee or below

No flare, straight silhouette

Details:

Very fitted, shows hip and thigh shape

Knee-length: ends just at or below knee

Slit (optional): Opening at back hem for movement (small triangle gap)

Minimal folds (fabric is taut)

DRESS (Various Styles):

FIT & FLARE DRESS:

Fitted bodice (bust and waist defined)

Flared skirt from waist or hips

Sleeveless, short, or long sleeves

SHEATH DRESS:

Straight, fitted throughout

Follows body closely

Professional appearance

Knee-length typical

Details Common to Dresses:

Bust definition: 

Curved line under bust (empire waist) OR

Darts from side toward bust point (fitted)

Shadow under bust line for dimension

Waist definition: 

Even if not belted, slight shadowing at natural waist

Fitted dresses show hourglass curve

Neckline: As described earlier (various styles)

Hem: Clean finish, can have slight folds/drape at bottom

OUTERWEAR:

BLAZER / SUIT JACKET (Professional):

Construction:

Lapels (large triangular folded collar)

Button closure (1-3 buttons)

Structured shoulders (padding implied)

Hip length (ends at hip bone)

Sleeves to wrist

Details:

Lapels: 

Two large triangular flaps fold from neckline down

Notch where lapel meets collar (angular cut)

Edge stitching visible (1px line along perimeter)

Buttonhole on left lapel for pin/flower

Buttons: 

Large (6-8px diameter)

1 button = modern style, 2 buttons = classic

Positioned at natural waist or slightly above

Pockets: 

Two chest pockets (upper) - flap style or welted

Two hip pockets (lower) - flap or slash style

Breast pocket: Small rectangular, left chest

Shoulder structure: 

Slight padding visible (straighter shoulder line than body)

Seam line where sleeve attaches

Vents: 

Opening at back hem (center or side)

Allows jacket to drape when sitting

Fold Placement:

Lapel fold line (diagonal crease)

Sleeve folds at elbow (3-4 lines)

Slight bunching at button closure

Back vent shows triangular gap when walking

CARDIGAN (Casual Jacket):

Covered earlier under sweaters

Button-front, no lapels

Softer, knit fabric

More casual than blazer

COAT (Heavy Outerwear):

Construction:

Long length (knee or mid-calf)

Heavy fabric appearance

Large buttons or zipper

Collar, lapels, or hood

Details:

Very few folds (heavy, stiff fabric)

Large shadows (thick material creates depth)

Belt optional (wrapped around waist)

Deep pockets (large rectangular flaps)

👗 CULTURAL & TRADITIONAL CLOTHING:

INDIAN WEAR (Salwar Kameez - Example):

KAMEEZ (Tunic Top):

Construction:

Long tunic, thigh-length or knee-length

Short sleeves or 3/4 sleeves

Loose fit, comfortable

Side slits from hem up 15-20cm

Details:

Neckline: 

Keyhole neckline (small teardrop opening at chest center)

Embroidered edge (tiny decorative marks around neckline, 0.5px)

Piping in contrasting color (thin outline, 1px)

Sleeves: 

Straight cut, not fitted

Hem can have decorative border

Side slits: Allow movement, show salwar underneath

Embroidery (optional): 

Along neckline, cuffs, hem

Simple line patterns, floral motifs

Contrasting thread color

SALWAR (Pants):

Construction:

Loose, baggy pants

Gathered at waist with drawstring

Tapered and gathered at ankle

Details:

Waist: Many small vertical folds (gathered fabric)

Ankles: Cuff or gathered band

Loose through thighs and knees

Vertical fold lines throughout

DUPATTA (Scarf/Shawl):

Construction:

Long rectangular fabric (2-3 meters)

Draped over one or both shoulders

Can be wrapped, pleated, or flowing

Details:

Drapes diagonally across body (shoulder to opposite hip)

Falls in graceful folds (3-5 major drape lines)

Ends hang loose or tucked

Semi-transparent optional (see body outline underneath)

Decorative border at edges

OTHER CULTURAL GARMENTS:

Principles for rendering ANY cultural clothing:

Research authentic construction and draping

Show proper layering and fabric behavior

Include culturally specific details (embroidery, closures, accessories)

Respect accurate representation

Use appropriate color palettes for culture and occasion

👔 PROFESSIONAL UNIFORMS:

PRIEST / CLERGY:

CASSOCK (Long Robe):

Construction:

Floor-length robe

Button-front (10-15 small buttons down center)

Long sleeves with buttoned cuffs

Ankle-length hem

Details:

Clerical Collar: 

White band at neck (2-3cm tall)

Completely encircles neck

Small gap at front center (where it fastens)

Stark contrast against black cassock

Buttons: Small, closely spaced, black on black (subtle)

Cincture (Belt): 

Rope or fabric belt at waist

Can have knotted ends hanging

Optional: matches vestment color

Vertical folds: From shoulders down (4-6 major fold lines)

Cross necklace: 

Gold or silver chain

Cross pendant at chest level

Simple or ornate design

Vestments (Optional additions):

Stole: Long narrow scarf over shoulders, hanging down front 

Decorative border and symbols

Color varies by liturgical season (purple, green, white, red)

Chasuble: Poncho-like outer garment (for mass) 

Wide, drapes over shoulders and arms

POLICE / SECURITY OFFICER:

UNIFORM SHIRT:

Construction:

Short or long sleeve button-down

Epaulettes on shoulders (rectangular flaps with buttons)

Badge on left chest

Name tag on right chest

Details:

Badge: 

Shield or star shape

Metallic appearance (silver or gold)

Positioned 5cm below collar on left

Highly reflective (white highlight on top edge)

Epaulettes: 

Shoulder straps

Buttoned at shoulder point

May have rank insignia (stripes, bars)

Chest pockets: 

Two large patch pockets with flaps

Button closure on flaps

Pen holder loops visible

Patches: 

Shoulder patches (unit/department insignia)

Embroidered design, circular or shield

Color contrast with uniform

DUTY BELT:

Construction:

Wide leather belt (5-6cm)

Multiple equipment pouches attached

Details:

Belt buckle: Large, rectangular, metallic

Equipment (simplified rendering): 

Radio holster (rectangular black box, left hip)

Baton holder (cylindrical, right hip)

Handcuff case (small rectangular, back)

Magazine pouches (small rectangles)

ALL simplified as basic shapes, 2px outlines

Black leather, matte finish

PANTS:

Solid color (black, navy, tan)

Cargo pockets optional (large rectangular flap pockets on thighs)

Crease line down center

Boots: Black tactical boots, laced

MEDICAL PROFESSIONAL (Doctor/Nurse):

LAB COAT / WHITE COAT:

Construction:

Knee-length white coat

Button or snap closure

Large collar

Long sleeves

Details:

Pockets: 

Two large patch pockets at hip level

One chest pocket (left side)

Pockets often contain pens, medical tools (visible tops)

Collar: Can be worn up or folded down

Sleeves: Can be rolled up to forearm

Name badge: Clipped to chest pocket

Stethoscope: Draped around neck (black rubber tubing, metallic chest piece)

SCRUBS (Alternative):

Simple V-neck tunic top

Drawstring pants

Solid colors (blue, green, burgundy, navy)

Minimal details, very functional

MECHANIC / WORKER:

COVERALLS / OVERALLS:

Construction:

Full-body garment OR

Bib-style overalls over shirt

Details (Overalls):

Bib: Large rectangular chest panel

Straps: Over shoulders, adjustable buckles 

Metal buckles (silver rectangular shape)

Cross in X-pattern on back OR parallel

Chest pocket: On bib panel, large 

Flap closure

Tools visible (screwdriver, pen)

Multiple pockets: 

Hip pockets

Thigh pockets (cargo style)

Hammer loop (small loop on side)

Knee reinforcement: Darker patch fabric at knees

Heavy fabric: Denim or canvas, thick appearance

Grease stains: 

Dark smudges on knees, thighs, chest

Brown-black color (#2A1A0A)

Irregular shapes, 50-70% opacity

WORK SHIRT (underneath or alone):

Long sleeve button-down

Chest pockets with flap

Name patch (oval or rectangular, embroidered name)

Company logo patch on shoulder or chest

Colors:

Navy blue, grey, tan, olive green

High-visibility: Orange or yellow with reflective strips

✅ CLOTHING CONSTRUCTION CHECKLIST:

Every garment must have: ✅ Appropriate fit for body type
✅ Correct fabric draping and folds
✅ Proper layering (visible edges of underlayers)
✅ Shading consistent with body shading
✅ Details appropriate to garment type
✅ Color palette harmonious
✅ Outlines clean and professional
✅ Wrinkles placed logically at stress points

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 6 OF 10

💍 ACCESSORIES, JEWELRY, FOOTWEAR & EQUIPMENT SYSTEMS

🎯 ACCESSORY PHILOSOPHY & PRINCIPLES:

Core Accessory Approach:

Accessories add character personality, profession, culture

Must be proportionally accurate to body scale

Detail level: High for visible items, simplified for background

Functional realism (items worn correctly, positioned naturally)

Strategic placement (don't overcrowd character)

Support character narrative (priest has cross, worker has tools)

Accessory Hierarchy:

Primary: Essential to character identity (doctor's stethoscope, priest's collar)

Secondary: Enhancement items (jewelry, watches, bags)

Tertiary: Optional details (pins, badges, minor decorations)

Limit accessories: 3-5 items maximum per character (avoid visual clutter)

👟 FOOTWEAR SYSTEMS:

MEN'S FORMAL SHOES:

OXFORD / DERBY SHOES (Dress Shoes):

Construction:

Lace-up closure at center front

Low heel (2-3cm)

Pointed or rounded toe

Leather appearance

Details:

Toe cap: Horizontal line across toe area (indicates stitched cap)

Laces: 

Zigzag pattern, 1px lines

4-6 crossing points visible

Shoelace holes (eyelets) as small circles (2px)

Heel: 

Defined rectangular block at back

Shadow underneath heel for elevation

2px outline separating heel from sole

Sole edge: 

1.5-2px line separating upper from sole

Slight platform visible around perimeter

Shine/Highlight: 

Curved white highlight on toe area (1-2cm, 60% opacity)

Indicates polished leather surface

Positioned top-center of shoe

Color:

Black (#1A1A1A): Most formal

Dark brown (#3A2410): Business casual

Tan (#8B6A48): Casual formal

Shading:

Base color + darker shadow (30% darker) on sides and heel

Highlight on top center (toe and instep)

LOAFERS (Slip-On Formal/Casual):

Construction:

No laces, slip-on design

Low or no heel

Clean, simple silhouette

Details:

Smooth upper, no lacing

Optional: Metal bit/buckle detail across instep (small rectangle, metallic shine)

Moccasin stitching visible (U-shaped line around toe area, 1px)

Similar shading to oxfords

MEN'S CASUAL SHOES:

SNEAKERS / ATHLETIC SHOES:

Construction:

Lace-up closure

Thick sole (3-4cm)

Padded collar around ankle

Sporty, casual appearance

Details:

Sole: 

Thick white or colored rubber sole

Clear separation from upper (2px line)

Can be two-tone (white sides, colored bottom)

Visible tread pattern optional (horizontal lines on sole edge)

Laces: 

Multiple crossings (6-8 points)

Can be colorful, contrasting

Panels: 

Multiple fabric/leather panels (3-5 sections)

Different colors for visual interest

Panel lines: 1.5px outlines

Logo area: 

Side panel or tongue can have simplified swoosh/brand mark

Don't render detailed logos (keep generic)

Toe cap: Rounded, reinforced appearance

Collar padding: Puffy appearance at ankle opening

Color Schemes:

Classic: White upper, white sole (#F5F5F5)

Athletic: White + colored accents (blue, red, green panels)

Black: All-black sneakers (#2A2A2A)

BOOTS (Work/Casual):

Construction:

Extends above ankle (mid-calf or higher)

Lace-up front or side zipper

Thick sole, defined heel

Rugged appearance

Details:

Shaft: 

Tall section above ankle (15-30cm)

Slight wrinkles/creases at ankle flex point

Can have strap/buckle detail

Laces: 

Many crossings (8-12) up the shaft

Thick laces, prominent

Sole: 

Very thick (4-5cm), chunky

Deep tread visible as horizontal lines on edge

Toe: 

Reinforced cap (horizontal line)

Rounded or square toe

Heel: Block heel, substantial

Types:

Work boots: Tan/brown leather, heavy-duty

Combat boots: Black, military style, many laces

Chelsea boots: Elastic side panels, no laces, ankle height

WOMEN'S FORMAL SHOES:

HIGH HEELS / PUMPS:

Construction:

Closed toe (covered)

Stiletto or block heel (7-12cm tall)

Slip-on or strap closure

Elegant, formal appearance

Details:

Heel: 

CRITICAL: Thin stiletto (2-3px width at top, 1-2px at bottom)

OR block heel (rectangular, 4-6px width constant)

Positioned at back of foot

Shadow cast by heel on ground

Toe: 

Pointed (triangular toe box) OR

Rounded (curved toe box) OR

Peep-toe (open at tip showing toes)

Sole: 

Thin, follows foot curve

Minimal platform

Upper: 

Smooth leather or fabric

Clean lines, elegant curves

Shine: 

Glossy highlight on toe area (patent leather effect)

50-70% opacity white streak

Color:

Black (#1A1A1A): Classic, versatile

Nude/Beige (#C8A078): Elegant, leg-lengthening

Red (#B82020): Bold, statement

Metallic: Silver (#C8C8C8) or gold (#D4A858)

FLATS (Ballet Flats, Loafers):

Construction:

No heel or minimal heel (1cm)

Slip-on design

Simple, comfortable

Details:

Rounded toe

Elastic or strap at sides (keeps shoe on)

Bow detail optional (small decorative bow at toe center, 1cm)

Minimal structure, follows foot shape

Can have toe cap line or panel details

WOMEN'S CASUAL SHOES:

SNEAKERS:

Similar to men's but often more streamlined

Slimmer sole (2-3cm)

Can have platform sole (chunky trend)

Pastel or bright colors common

SANDALS:

Open toe and heel

Straps across foot (1-3 straps)

Flat or low wedge heel

Summer, casual appearance

ANKLE BOOTS:

Similar to men's but more fitted to leg

Heel common (block or stiletto)

Side zipper typical

Fashion-forward details (buckles, studs)

💍 JEWELRY & PERSONAL ACCESSORIES:

EARRINGS:

HOOP EARRINGS (Common for Women):

Construction:

Perfect circle or oval

1.5-2px outline

Metallic appearance

Sizes:

Small: 8-10mm diameter (subtle)

Medium: 15-20mm diameter (noticeable)

Large: 30-40mm diameter (statement piece)

Details:

Gold color: #D4A858 base, #F0C878 highlight

Silver color: #C8C8C8 base, #E8E8E8 highlight

Shine: Small white highlight at top curve (2-3px)

Positioning: Attached to earlobe, hangs vertically

STUD EARRINGS:

Construction:

Small circular or geometric shape

Sits on earlobe, doesn't dangle

3-5px total size

Details:

Simple circle or square (1.5px outline)

Metallic or gemstone appearance

Highlight in center (1-2px white dot)

Very subtle, refined

DANGLING EARRINGS:

Construction:

Decorative element hangs from earlobe

Connected by small chain or wire (0.5-1px line)

Movement implied

Details:

Top part: Small circle on earlobe

Connecting element: Thin line (1-2cm long)

Bottom element: Decorative shape (teardrop, geometric, charm)

Can have multiple tiers (2-3 elements stacked)

NECKLACES:

CHAIN NECKLACE (Simple):

Construction:

Thin line encircling neck

Can have small pendant

Details:

Chain: 0.5-1px curved line following neck curve

Clasp at back (not usually visible in front view)

Hangs 2-5cm below collar depending on length

Gold or silver color

Subtle shine segments (small highlights every 1-2cm, suggests chain links)

CROSS NECKLACE (Religious):

Construction:

Chain as above

Cross pendant at chest center

Details:

Cross: 

Vertical bar (1.5-2cm) intersected by horizontal bar (1-1.5cm)

Simple or ornate design

Gold, silver, or wood appearance

1.5px outline for cross shape

Optional: Small decorative details at ends

Positioning: Hangs at sternum level

Creates focal point on chest

PENDANT NECKLACE:

Construction:

Chain with decorative pendant

Details:

Pendant shapes: Circle, heart, rectangle, custom

Size: 1-2cm

Can have gemstone center (colored circle with highlight)

Metallic frame around gemstone

CHOKER:

Construction:

Tight band around neck

No hanging length, fits snug

Details:

3-5mm wide band

Solid color or pattern

Can be velvet (matte, black) or metallic

Optional: Small pendant at center front

BRACELETS & WATCHES:

WATCH (Wristwatch):

Construction:

Band around wrist

Watch face on top of wrist

Details:

Band: 

Leather strap (textured, brown/black) OR

Metal links (segmented, silver/gold)

Wraps around wrist, clasp at underside

Width: 2-3cm on wrist

Watch Face: 

Circular or rectangular (1-1.5cm)

Light colored face (#F5F5F5)

Two hands visible: hour and minute (thin black lines)

Optional: Numbers around edge (tiny, 0.5px)

Glass surface: Curved white highlight (50% opacity)

Metallic rim: Gold or silver outline (1.5px)

Positioning: On left wrist typically, band visible on underside

BRACELET (Simple):

Construction:

Band or chain around wrist

Decorative or minimal

Details:

Thin chain: 1px line with small link suggestions

Bangle: Solid band, 3-5mm width, smooth

Charm bracelet: Chain with small hanging elements (simplified shapes)

Gold or silver color

RINGS:

Construction:

Small band on finger

Simple or decorative

Details:

Simple Band: 

Thin line around finger (1px)

Gold or silver color

Positioned on ring finger typically (fourth finger)

Slight shine on top (small highlight)

Gemstone Ring: 

Band as above

Small gemstone on top (2-3px circle or square)

Gemstone has highlight (1px white dot)

Color: Diamond (clear/white), ruby (red), sapphire (blue), emerald (green)

Positioning: Most commonly on ring finger, but can be on any finger

👓 EYEWEAR:

GLASSES (Prescription/Reading):

FRAME CONSTRUCTION:

Rectangular Frames:

Two rectangular lens shapes

Bridge connecting center (2-3px)

Temple arms extending to ears

Details:

Frames: 

Thick outline: 3-4px black or colored lines

Rounded corners on rectangles (not sharp 90°)

Bridge: Small rectangular or curved connector (2px wide)

Nose pads: Two small ovals where bridge meets face (optional detail)

Temple Arms: 

Extend from frame corners horizontally back

Slightly curve down behind ears

Same color/thickness as frames

Visible from front view extending to side of head

Lenses: 

Completely transparent (NO visible glass)

Eyes and eyebrows fully visible through lenses

NO reflections or glare (flat, invisible)

CRITICAL: Lenses are just empty space within frame outline

Frame Styles:

Rectangle: Classic, professional (frames described above)

Round: Circular lens shapes, vintage/intellectual look

Cat-eye: Upswept outer corners, feminine, retro

Aviator: Teardrop shapes, metal thin frames

Colors:

Black (#1A1A1A): Most common, professional

Tortoiseshell: Brown mottled pattern (#6B4A28 + #3A2410)

Metal: Silver (#C8C8C8) or gold (#D4A858), thin frames (2px)

Colored plastic: Bold colors (red, blue, green) for personality

Shadow Detail:

Slight shadow cast on cheeks below frames (very subtle, 10% opacity)

Small shadow where bridge rests on nose

SUNGLASSES:

Construction:

Same frame structures as glasses

Lenses are OPAQUE, dark

Details:

Lenses: 

Solid dark color (black, dark grey, brown)

Completely block view of eyes

Flat color fill, NO transparency

Slight curved highlight on upper lens area (suggests glossy surface, 1-2cm white streak, 30% opacity)

Frames: Same as prescription glasses

Can be worn on head (pushed up on forehead) instead of over eyes

Styles:

Aviator (metal, teardrop)

Wayfarer (thick plastic, rectangular)

Round (vintage, small circles)

Sport (wraparound, futuristic)

🎒 BAGS & CARRIED ITEMS:

HANDBAG / PURSE (Women):

Construction:

Small rectangular or curved shape

Handle or strap

Held in hand or on arm

Details:

Size: 10-15cm visible portion

Leather texture (subtle horizontal lines, 0.5px)

Clasp or zipper at top (metallic accent)

Strap: Thin line (1-2px) if shoulder bag

Shadow underneath bag (grounding it)

Types:

Clutch: No strap, held in hand, small

Shoulder bag: Long strap, hangs at hip

Tote: Large, open top, handles

MESSENGER BAG / SATCHEL:

Construction:

Rectangular bag

Long strap worn across body (diagonal)

Unisex, professional or casual

Details:

Bag body: 

Rectangle at hip level (15-25cm visible)

Front flap with buckle or magnetic closure

Leather or canvas appearance

Color: Brown (#6B4A28), black (#2A2A2A), olive (#4A5A3A)

Strap: 

Wide strap (3-4cm), 2px outline

Crosses body diagonally (shoulder to opposite hip)

Visible on both sides of neck

Adjustable buckle detail optional

Same color as bag

Buckles/Hardware: 

Metallic silver or brass color

Rectangular or circular shape (5-8px)

Positioned on flap closure and strap

Positioning: Bag sits at side of hip, strap crosses chest

BACKPACK:

Construction:

Two shoulder straps

Bag on back (not visible in front view)

Straps visible over shoulders

Details (from front view):

Straps only visible: 

Two parallel straps over shoulders (2-3cm wide each, 2px outline)

Padding visible (slightly puffy appearance)

Color: Black, grey, navy typically

Buckles at chest level optional (sternum strap)

Bag body not visible (behind character)

Strap shadows fall on chest/shoulders

🔧 PROFESSIONAL EQUIPMENT & TOOLS:

MEDICAL STETHOSCOPE:

Construction:

Rubber tubing in U-shape around neck

Metal chest piece hanging at sternum

Details:

Tubing: 

Black rubber (matte finish, #2A2A2A)

3-4px diameter outline

Drapes naturally around neck

U or V shape

Earpieces: 

Two small angled pieces at top (one for each ear)

Metal or plastic, small (5-8px)

Chest piece: 

Circular metal disc (2-3cm diameter)

Silver or chrome color (#C8C8C8)

Bright highlight on top (white, 50% opacity)

Hangs at center chest on tubing

Positioning: Worn around neck, chest piece hangs at sternum or higher

TOOL BELT (Construction/Mechanic):

Construction:

Wide belt around waist

Multiple pouches and tool holders

Details:

Belt: 

Thick leather (5-7cm wide)

Brown or black

Large buckle (metallic, 2x3cm rectangle)

Pouches: 

3-5 rectangular pouches attached to belt

Various sizes (8-15cm wide)

Flap closures or open tops

Riveted construction (small metal dots at corners)

Tools (simplified, partially visible): 

Hammer handle protruding from loop

Screwdriver tops visible in pocket

Measuring tape clip

All rendered as simple shapes (outlines only, 1.5px)

Positioning: Around waist, over pants, tools hang at hip level

UTILITY VEST (Photographer, Worker):

Construction:

Sleeveless vest

Many pockets on front

Details:

Multiple patch pockets (6-12) in various sizes

Zippered pockets (thin zipper lines visible)

Velcro closures (rectangular flap tabs)

Neutral colors (tan, olive, black)

Worn over shirt

BADGE / ID CARD:

Construction:

Rectangular card in holder

Clipped to clothing or lanyard

Details:

Card: 

Rectangle (5cm x 8cm)

Photo area (small square, simplified face or solid color)

Text lines (3-5 horizontal lines representing text, 0.5px)

Company logo area (simplified geometric shape)

Holder: 

Clear plastic outline (1px)

Clip at top (metallic, simple shape)

Lanyard (if used): 

Thin strap around neck (1-2px)

Card hangs at chest level

Neutral color or branded pattern

Positioning: Clipped to chest pocket, belt, or lanyard around neck

💼 SPECIALIZED ACCESSORIES:

TIE (Necktie):

Construction:

Narrow fabric strip under collar

Knotted at neck, hangs down chest center

Details:

Knot: 

Triangular shape at collar (1.5-2cm wide)

Several fold lines suggesting knot structure (2-3 lines)

Blade (long part): 

Vertical strip (6-8cm wide)

Extends to waist or slightly below

Tapers to point at bottom (triangular end)

Patterns (optional): 

Solid color (most simple)

Diagonal stripes (thin parallel lines, 1px, 1-2cm apart)

Dots/polka dots (small circles scattered)

Keep pattern subtle, not overpowering

Shadow: 

Tie casts shadow on shirt underneath

Darker outline on right side (shadow edge)

Colors:

Professional: Navy, burgundy, grey, black

Formal: Black or white (with tuxedo)

Business: Blue, red (power colors)

BOW TIE:

Construction:

Butterfly shape at collar

Horizontal, centered on neck

Details:

Two triangular or rounded lobes (left and right)

Knot in center (small rectangular band)

Symmetrical shape

2px outline, same shading rules

Formal or quirky accessory

BELT:

Construction:

Band around waist

Buckle at front center

Details:

Band: 

Visible portion at front only (sides hidden by clothing)

3-5cm wide

Leather texture (subtle horizontal grain lines, 0.5px)

Holes visible (small circles, 1px, 2cm apart)

Buckle: 

Rectangular or square frame (3-4cm wide)

Metallic (silver, brass, gunmetal)

Prong visible (small rectangular tongue through hole)

Shine on top surface (highlight, 30% white)

Belt loops: 

Fabric loops on pants (5-8 around waist)

Belt threads through these (belt visible between loops)

Colors:

Black leather (#1A1A1A)

Brown leather (#3A2410, #6B4A28)

Match belt to shoes traditionally

SCARF:

Construction:

Long fabric strip wrapped around neck

Various wrapping styles

Details:

Width: 10-20cm visible

Drapes loosely or wrapped snugly

Can see both ends hanging (in front of body)

Fabric folds and layers (3-5 fold lines)

Patterns: Plaid, stripes, solid, floral

Materials: Wool (thick, winter), silk (thin, elegant), cotton (casual)

Styles:

Loop wrap: Once around neck, both ends hang in front

Infinity scarf: Continuous loop, no ends

Shawl: Wide, draped over shoulders

🎭 CULTURAL ACCESSORIES:

INDIAN JEWELRY:

Bindi (Forehead Decoration):

Small dot or decorative mark on forehead between eyebrows

Circular (2-4px diameter) or teardrop shape

Red, black, or decorative gemstone appearance

Positioned at "third eye" location

Bangles (Arm Bracelets):

Multiple thin bracelets stacked on wrist/forearm

3-10 bangles together

Metallic (gold, silver) or colored (glass bangles)

Each bangle: Thin circle outline (1px), 2-3mm spacing between

Nose Ring (Nath):

Small hoop or stud on nostril

Side of nose (left most traditional)

Tiny (2-3px) circle or dot

Gold or silver color

Maang Tikka (Headpiece):

Jewelry piece on forehead, attached to hair parting

Chain goes from hair down to forehead

Decorative pendant hangs on forehead

Intricate but simplified rendering

✅ ACCESSORY APPLICATION CHECKLIST:

Every accessory must: ✅ Be proportionally accurate to body
✅ Be positioned correctly (watches on wrists, badges on chest)
✅ Have appropriate detail level (visible items clear, background simplified)
✅ Follow shading rules (shadows cast appropriately)
✅ Support character narrative (profession, culture, personality)
✅ Not overcrowd (3-5 items maximum)
✅ Be rendered with clean linework (2px outlines standard)

Accessories must NEVER: ❌ Float unnaturally (must attach/rest logically)
❌ Be disproportionately large or small
❌ Clash with overall character style
❌ Obscure important character features
❌ Be overly detailed (keep simplified, clean)

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 7 OF 10

🎯 COMPLETE WORKFLOW, PROCESS SYSTEM & QUALITY CONTROL

📋 MASTER WORKFLOW OVERVIEW:

Complete Character Creation Process (10 Steps):

Character Analysis & Planning (Understanding requirements)

Base Sketch Construction (Proportions and pose)

Linework Execution (Clean outlines)

Base Color Application (Flat color fills)

Shading Implementation (Cel-shading system)

Detail Refinement (Features, textures, accessories)

Highlight Application (Selective brightness)

Final Line Cleanup (Edge perfection)

Quality Control Check (Verification against standards)

Export Preparation (Transparent background, resolution)

STEP 1: CHARACTER ANALYSIS & PLANNING

Input Processing:

User will provide character description containing:

Character Type: NORMAL or HORROR (CRITICAL classification)

Age Category: Kids, Young Adult, Middle Age, or Elderly

Gender: Male or Female (for anatomical accuracy)

Physical Description: 

Height/build (tall, short, slim, muscular, heavy)

Skin tone (light, medium, dark, or specific ethnicity)

Hair (color, length, style)

Eye color

Facial features (facial hair, glasses, scars, etc.)

Clothing: 

Outfit type (casual, formal, uniform, traditional)

Specific garments (shirt, pants, dress, etc.)

Colors and patterns

Accessories: Jewelry, bags, tools, equipment

Expression/Mood: Neutral, happy, serious, etc. (horror = emotionless default)

Optional Details: Specific cultural elements, profession, personality traits

Planning Phase Checklist:

✅ Identify Character Type:

If HORROR specified → Apply ALL horror rules from Part 4

If NORMAL or unspecified → Apply standard character rules

✅ Determine Proportional System:

Kids: 6-6.5 head heights, larger eyes, rounder face

Young Adult: 7.5-8 head heights, standard proportions

Elderly: 6.5-7 head heights, wrinkle lines, stooped posture

✅ Gender Anatomy Verification:

Male: Broader shoulders, straight waist-hip, flat chest

Female: Narrower shoulders, hourglass curves, bust definition

✅ Skin Tone Selection:

Reference Part 3 color palettes

Normal: Vibrant, healthy tones

Horror: Desaturated grey/blue/green tones

✅ Clothing Style Planning:

Reference Part 5 garment specifications

Plan layering order (underwear → shirt → jacket → accessories)

Select appropriate colors and details

✅ Accessory Integration:

Maximum 3-5 accessories

Ensure functional placement

Support character narrative

STEP 2: BASE SKETCH CONSTRUCTION

Foundation Building:

Proportional Framework:

Head Unit Establishment:

Draw head oval (represents 1 unit)

Mark centerline (vertical symmetry axis)

Mark horizontal eye line (center of head)

Body Height Mapping:

Mark total height (7.5-8 heads for adult)

Divide into segments: 

Head: 0-1

Neck/Shoulders: 1-1.5

Torso: 1.5-4

Hips: 4-4.5

Thighs: 4.5-6

Calves: 6-7.5

Feet: 7.5-8

Width Guidelines:

Shoulders: 2.5 heads (male) or 2 heads (female)

Hips: 2 heads (male) or 2.5 heads (female)

Waist: 2 heads (male) or 1.5 heads (female)

Skeletal Structure:

Simple stick figure with joint circles

Spine line (straight, vertical for front view)

Shoulder line (horizontal, level)

Hip line (horizontal, level)

Arm and leg attachment points

Volume Addition:

Add geometric shapes for body masses

Head: Oval

Torso: Rectangle or trapezoid

Arms: Cylinders, tapering to wrists

Legs: Cylinders, tapering to ankles

Hands: Mitten shapes

Feet: Simplified shoe shapes

Pose Verification:

✅ Front-Facing Alignment:

Body perfectly centered on vertical axis

Left and right sides symmetrical

Shoulders level (parallel to ground)

Hips level (parallel to ground)

Feet positioned at shoulder width or narrower

Head facing forward (not tilted or turned)

Eyes looking at viewer (direct gaze)

✅ Posture Check:

Spine straight (vertical line)

Weight evenly distributed on both feet

Arms hanging naturally at sides

Hands visible, fingers relaxed

No leaning, twisting, or dynamic action

Static, standing position

STEP 3: LINEWORK EXECUTION

Clean Outline Construction:

Linework Order (Outside to Inside):

Outer Silhouette First:

Head outline (2.5-3px)

Body outline (2.5-3px)

Arms outline (2.5-3px)

Legs outline (2.5-3px)

Complete outer boundary before interior

Major Divisions:

Neck separation (2px)

Clothing boundaries (2-2.5px)

Limb segments (2px)

Facial Features:

Eyes (1.5-2px outlines, irises, pupils)

Eyebrows (solid shapes, 2-4px height)

Nose (1-1.5px minimal lines)

Mouth (1.5-2px)

Ears (2px outer, 1px inner detail)

Hair Construction:

Major hair chunks (2-3px outlines)

Hair separation lines (1-1.5px)

Highlight stroke paths (1-2px, will be colored later)

Clothing Details:

Seams (1.5-2px)

Folds (1-1.5px)

Buttons (1px circles)

Pockets (1.5px)

Collars, cuffs, hems (2px)

Accessories:

Glasses frames (3-4px)

Jewelry (1.5-2px)

Bags (2px outlines)

Equipment (2px)

Line Quality Standards:

✅ Vector-Clean Paths:

Smooth curves (no jagged edges)

Sharp corners where appropriate (clothing angles)

Rounded corners where appropriate (organic forms)

Continuous lines (no gaps in outlines)

✅ Consistent Weight:

Same thickness maintained along entire path

No accidental thinning or thickening

Deliberate weight hierarchy followed

✅ Clean Intersections:

Lines meet precisely at corners

No overlapping double lines

T-junctions clean (one line stops at other)

✅ Closed Shapes:

All color regions fully enclosed

No gaps in outlines (prevents color leaks)

Proper path closure for fill areas

STEP 4: BASE COLOR APPLICATION

Flat Color Fill System:

Color Application Order:

Skin Tone (Foundation):

Fill all visible skin areas with base skin color

Face, neck, hands, any exposed arms/legs

One uniform color (no shading yet)

Reference Part 3 for exact color codes

Hair Base Color:

Fill all hair areas with base hair color

Single flat color across entire hair mass

No highlights or shadows yet

Eye Components:

Sclera (white): #FAFAFA or #F5F5F5

Iris: Base color (brown, blue, green, etc.)

Pupil: Pure black #000000

Clothing Base Colors:

Fill each garment with its base color

Work from innermost layer to outer: 

Underwear/undershirt (if visible)

Main shirt/top

Pants/skirt/dress

Jacket/coat/outer layer

Each garment one solid color initially

Accessory Base Colors:

Shoes: Base color

Bags: Base color

Jewelry: Gold (#D4A858) or silver (#C8C8C8)

Equipment: Base colors

Detail Elements:

Lips: Base lip color

Tongue (if mouth open): Pink-red

Teeth (if visible): Off-white #F0E8D8

Fingernails (if visible): Pink or painted color

Color Separation Verification:

✅ No Color Bleeding:

Each area confined to its outline

Clean boundaries between colors

No overlapping fills

✅ Complete Coverage:

No white gaps inside shapes

All areas within outlines filled

Outlines not covered by fill (lines on top layer)

✅ Layer Organization:

Skin on skin layer

Hair on hair layer

Each clothing item on separate layer

Accessories on accessory layer

Organized for easy editing

STEP 5: SHADING IMPLEMENTATION

Cel-Shading Application:

Shadow Creation Process:

Determine Light Source:

Standard: Top-front-left at 45° angle

Consistent for entire character

Never changes mid-character

Shadow Color Calculation:

For each base color, create shadow color

Formula: 25-35% darker than base

Normal characters: 25-30% darker

Horror characters: 35-45% darker

Maintain same hue, only darken value

Shadow Shape Design:

Create organic curved shapes

Follow anatomical form (crescents, ovals, strips)

Hard vector edges (no blur, no gradients)

Cover 30-40% of base area (normal) or 40-60% (horror)

Shadow Placement Guide:

FACE:

Right side of nose (thin crescent)

Under cheekbones (curved strip)

Under lower lip (small crescent)

Under jaw/chin (following jawline)

Right side of face (temple to jaw)

Eye sockets (very subtle for normal, deep for horror)

Under eyebrows (subtle shadow line)

NECK:

Right side

Under chin (cast shadow from head)

Sides of neck cylinder

TORSO:

Right side of body

Under bust (female) - curved shadow

Sides of waist

Under any overlapping clothing layers

ARMS:

Right side of each arm

Inner arm (underside)

Armpits (deep shadow)

Under bent elbows

LEGS:

Right side of each leg

Inner thighs

Behind knees

Under bent knees

CLOTHING FOLDS:

Valley of each fold (recessed area)

Both sides of crease line get shadow

Under collar, lapels

Under pocket flaps

Inside sleeve openings

HAIR:

Under overlapping chunks

At scalp/roots

Behind ears

Back layers (depth)

Between major segments

ACCESSORIES:

Under bag flaps

Under jewelry (cast shadows on skin/clothes)

Under glasses frames (on cheeks)

Sides away from light

Shadow Application Technique:

Create Shadow Shape:

Draw vector path defining shadow area

Use pen tool for clean curves

Ensure hard edges (no feathering)

Fill with Shadow Color:

Apply calculated shadow color

Completely opaque (100% opacity)

No transparency or gradients

Shadow Layering:

Shadows on top of base colors

Below line art layer

Each shadow on its parent color layer

Consistency Check:

All shadows from same light direction

Shadow darkness consistent across character

Shadow shapes logical and anatomical

STEP 6: DETAIL REFINEMENT

Adding Textures and Fine Details:

Facial Details:

✅ Eye Highlights:

Small white dot or oval in iris/pupil (3-5px)

Upper-left position typically

Pure white #FFFFFF

Creates "life" in eyes

OMIT for horror characters (dead stare)

✅ Lip Detail (Female):

Center of lower lip: Lighter tone (highlight)

Corners of mouth: Darker tone

Two-tone rendering for dimension

✅ Facial Hair Texture (Male):

Mustache: Individual brush strokes

Beard: Short line marks for texture

Stubble: Tiny dots or dashes

Shadow underneath for depth

✅ Wrinkles (Elderly):

Forehead: 3-4 horizontal lines (1px)

Crow's feet: 3-5 radiating lines from outer eye

Nasolabial folds: Deep curved lines nose to mouth

Under eyes: Bags (shadow shapes)

Mouth: Vertical lines above upper lip

✅ Skin Details (Horror):

Veins: Thin blue-purple lines (0.5-1px, 40% opacity)

Discoloration patches: Irregular darker areas

Under-eye circles: Heavy purple-brown shadows

Mottled texture: Subtle color variation zones

Hair Details:

✅ Highlight Strokes:

3-8 curved strokes per major hair chunk

1-2px width, following hair flow

Lighter color (15-25% brighter than base)

Steel-blue for black hair (#4A6B7C)

Positioned on "top" surface of chunks

✅ Hair Depth:

Front chunks: Lighter overall

Back chunks: Darker shadows

Creates volume through value

Clothing Details:

✅ Fabric Texture (Subtle):

Ribbed knits: Horizontal lines (0.5px, 1-2mm apart)

Denim stitching: Double parallel lines, gold color

Leather grain: Subtle horizontal texture lines

Worn areas: Slightly darker patches

✅ Stains (Horror Clothing):

Small irregular shapes (5-15mm)

Murky brown-grey or dark red-brown

50-70% opacity over fabric

1-3 stains maximum

Mysterious, ambiguous appearance

✅ Wear Indicators:

Compressed areas: Slightly darker (shoulders, elbows, seat)

Edges slightly frayed: Tiny texture marks

Buttons misaligned or missing

Accessory Details:

✅ Metallic Shine:

Jewelry: Curved white highlight on top surface

Belt buckle: Bright highlight (30-50% opacity)

Watch face: Glass glare (curved white streak)

Buttons: Small center highlight dot

✅ Leather Texture:

Shoes: Curved highlight on toe area

Bags: Subtle horizontal grain lines

Belts: Slight texture marks

✅ Glass/Transparent Materials:

Glasses lenses: Completely clear (see through)

Watch crystal: Curved highlight only

NO reflections or complex rendering

STEP 7: HIGHLIGHT APPLICATION

Selective Brightness (Use Sparingly):

Highlight Locations:

✅ Skin (Normal Characters Only):

Forehead center: Small soft area (5-10% lighter)

Cheekbone tops: Subtle brightness

Bridge of nose: Thin vertical highlight

Chin: Small center highlight

Horror characters: Minimal or NO skin highlights

✅ Hair:

Already covered in Step 6 (highlight strokes)

Most prominent highlight area

✅ Clothing (Selective):

Shoulders: Slight brightness on top surface

Smooth fabrics only: Very subtle sheen

Matte fabrics: NO highlights

✅ Accessories:

Metallic items: Bright highlights (jewelry, buckles)

Glossy items: Curved shine streaks (shoes, bags)

Glassware: Strategic glare marks

Highlight Technique:

Color: Base color + 10-20% lighter value

Coverage: Small areas only (5-15% of surface)

Edges: Can be slightly softer than shadows (subtle fade)

Opacity: 60-100% depending on material

Restraint: Less is more - avoid over-highlighting

STEP 8: FINAL LINE CLEANUP

Outline Perfection:

✅ Line Inspection:

Zoom to 200-400% and inspect every line

Check for gaps, overlaps, inconsistencies

Verify weight hierarchy maintained

Ensure all corners clean and precise

✅ Edge Refinement:

Smooth any rough curves

Sharpen angles where appropriate

Clean up any stray pixels

Perfect all intersections

✅ Line Color Verification:

Outlines pure black #000000 (except hair highlights)

No accidental grey or colored lines

Consistent throughout character

✅ Closure Check:

All shapes fully enclosed

No gaps in critical outlines

Color areas properly contained

STEP 9: QUALITY CONTROL CHECK

Comprehensive Verification:

MANDATORY CHECKLIST - ALL MUST PASS:

A. TECHNICAL REQUIREMENTS:

✅ Resolution: Minimum 3000x4000px at 300 DPI
✅ Background: Fully transparent (alpha channel, PNG format)
✅ View: Perfect front-facing, zero angle deviation
✅ Posture: Straight, centered, symmetrical
✅ Proportions: Correct for age category and gender
✅ Lines: Clean vector-quality, consistent weights
✅ Colors: Proper palette for character type (normal vs horror)

B. ANATOMICAL ACCURACY:

✅ Body Proportions:

Correct head height ratio (6-8 heads depending on age)

Proper shoulder width for gender

Accurate waist-hip ratio for gender

Arms reach mid-thigh

Legs are 50% of total height

✅ Facial Proportions:

Eyes at horizontal center of head

Eye spacing = 1 eye width apart

Nose and mouth positioned correctly

Ears from eyebrow to nose base

✅ Gender Differentiation (CRITICAL):

Male: Broad shoulders, straight waist-hip, flat chest, angular jaw

Female: Narrower shoulders, hourglass curve, bust definition, softer jaw

✅ Age Indicators:

Kids: Rounder face, larger eyes, fewer details

Young Adult: Standard proportions, smooth skin

Elderly: Wrinkle lines, thinner features, slight stoop

C. STYLE CONSISTENCY:

✅ 2D Flat Aesthetic: No 3D rendering, depth simulation
✅ Cel-Shading Only: Hard edge shadows, no gradients
✅ Line Quality: Professional vector-style throughout
✅ Color Harmony: Palette cohesive and appropriate
✅ Detail Balance: Not over-detailed, not under-detailed

D. CHARACTER TYPE VERIFICATION:

IF NORMAL CHARACTER: ✅ Warm, healthy skin tones (peach, brown, tan)
✅ Eyes have catch light highlights
✅ Natural, approachable expression
✅ Clean, well-maintained appearance
✅ Vibrant color palette

IF HORROR CHARACTER: ✅ Desaturated grey/blue/green skin tones
✅ Eyes have NO catch light (dead stare)
✅ Emotionless or subtly wrong expression
✅ Facial asymmetry present (subtle)
✅ Under-eye darkness pronounced
✅ Skin mottling/uneven tone
✅ Hair unkempt, heavy
✅ Clothing dark, worn, with ambiguous stains
✅ Dramatic shadows (40-60% coverage)
✅ NO obvious gore or monsters

E. CLOTHING & ACCESSORIES:

✅ Clothing fits body appropriately
✅ Fabric draping realistic (gravity observed)
✅ Folds placed logically at stress points
✅ Layering order correct
✅ Accessories positioned functionally
✅ Maximum 3-5 accessories (not overcrowded)
✅ Details support character narrative

F. PROFESSIONAL STANDARDS:

✅ Appears hand-crafted by professional artist
✅ Zero indication of AI generation
✅ Adobe Illustrator/Photoshop quality
✅ Suitable for professional publication
✅ Could pass as mobile game character art
✅ Print-ready quality

FAILURE POINTS - IF ANY OCCUR, CHARACTER MUST BE REVISED:

❌ 3D appearance or volumetric rendering
❌ Gradients in shading
❌ Angled or tilted view (not perfect front-facing)
❌ Background present (not transparent)
❌ Gender anatomy incorrect or ambiguous
❌ Proportions significantly off
❌ Line quality sketchy or rough
❌ Color palette wrong for character type
❌ Horror character has obvious monster features
❌ Normal character looks sickly or wrong
❌ Over-detailed or under-detailed
❌ Accessories incorrectly positioned

STEP 10: EXPORT PREPARATION

Final Output:

✅ File Format: PNG with full alpha transparency
✅ Resolution: Maintain 3000x4000px minimum at 300 DPI
✅ Background: Verify complete transparency (no white, no color)
✅ Layers: Can be flattened for export (all merged except background)
✅ Color Mode: RGB (for digital display)
✅ Compression: None or lossless (maintain quality)

✅ Visual Check:

View against different colored backgrounds to verify transparency

Check all edges clean (no fringing or halos)

Verify no artifacts or quality degradation

Confirm character centered in canvas

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 8 OF 10

📝 EXAMPLE PROMPTS, VARIATIONS & PRACTICAL APPLICATIONS

🎯 EXAMPLE PROMPT STRUCTURE:

Standard User Input Format:

CHARACTER TYPE: [Normal/Horror] AGE: [Kids/Young Adult/Elderly] GENDER: [Male/Female] ETHNICITY/SKIN TONE: [Specific description] HAIR: [Color, length, style] EYES: [Color] FACIAL FEATURES: [Glasses, facial hair, scars, etc.] EXPRESSION: [Neutral, smiling, serious, etc.] CLOTHING: [Detailed description] ACCESSORIES: [List of items] SPECIAL NOTES: [Any specific requirements] 

📋 EXAMPLE 1: NORMAL YOUNG ADULT MALE (Professional)

USER INPUT:

CHARACTER TYPE: Normal AGE: Young Adult (28 years) GENDER: Male SKIN TONE: Medium beige (Indian/South Asian) HAIR: Short black hair, neat professional style EYES: Dark brown FACIAL FEATURES: Clean shaven, no glasses EXPRESSION: Neutral, confident CLOTHING: - White formal dress shirt, long sleeves - Navy blue tie with subtle diagonal stripes - Grey dress pants with center crease - Black leather oxford shoes - Black leather belt with silver buckle ACCESSORIES: - Silver wristwatch on left wrist - Small black leather messenger bag (cross-body) SPECIAL NOTES: Professional businessman appearance 

EXECUTION BREAKDOWN:

Proportions:

Height: 8 head units (tall, professional stature)

Shoulders: 2.8 head widths (athletic build)

Straight posture, confident stance

Skin Tone Application:

Base: #D4A876 (golden tan)

Shadow: #B88858 (warm brown shadow)

Highlight: #E0C090 (light caramel highlight on forehead, nose bridge, chin)

Facial Details:

Eyes: Dark brown iris #5D3A1A, standard size, WITH catch light (normal character)

Eyebrows: Thick, black, straight masculine shape (5px height)

Nose: Minimal line construction, triangular shadow beneath

Mouth: Thin closed lip, neutral expression (#C8857A lip color)

Ears: Standard C-shape, visible both sides

Hair Rendering:

Base color: #1A1A1A (deep black)

Style: 8-10 chunky segments, short professional cut, swept to side

Highlights: Steel blue streaks #4A6B7C on top surface (4-5 strokes)

Neat, groomed appearance (not messy)

Clothing Construction:

White Dress Shirt:

Base: #FFFFFF (pure white)

Shadow: #E8E8E8 (light grey on sides, under armpits)

Collar: Pointed, 2px outline, crisp and clean

Buttons: 6 small white circles down center (3px diameter)

Cuffs: Visible at wrists, single button each

Tucked into pants at waist

Slight wrinkles at elbows (3-4 fold lines each)

Navy Tie:

Base: #2A3A5A (navy blue)

Shadow: #1A2A4A (darker navy on right side)

Knotted at collar, triangular knot (1.5cm wide)

Diagonal stripes: Thin silver lines 1px, 2cm apart, 45° angle

Extends to waist level, pointed triangular end

Grey Dress Pants:

Base: #6B6B6B (medium grey)

Shadow: #4A4A4A (charcoal grey on sides)

Center crease line: Sharp vertical line down each leg (1.5px)

Belt loops: 6 small rectangles at waist (1px outline)

Side pocket: Diagonal line from waist toward center

Slight fold at ankles above shoes

Professional, tailored fit

Black Belt:

Visible at front waist between shirt and pants

Width: 4cm

Color: #1A1A1A (black leather)

Silver rectangular buckle: #C8C8C8 with white highlight

Black Oxford Shoes:

Base: #1A1A1A (polished black)

Shadow: #000000 (pure black on sides and heel)

Toe cap line: Horizontal line indicating stitched cap

Laces: Zigzag pattern, 5 crossings visible (1px lines)

Shine highlight: Curved white streak on toe area (50% opacity, 2cm)

Positioned shoulder-width apart, slight V-stance

Accessories:

Silver Watch (Left Wrist):

Band: Silver metallic links, wraps around wrist (2px outline)

Watch face: Circular, 1.2cm diameter, white face #F5F5F5

Two black hands showing time (thin 0.5px lines)

Glass surface: Curved white highlight (40% opacity)

Metallic rim: Silver #C8C8C8

Messenger Bag:

Position: Diagonal strap from right shoulder to left hip

Strap: Brown leather #6B4A28, 3cm wide visible (2px outline)

Bag body: At left hip, brown leather rectangle (15cm visible)

Front flap with brass buckle closure (small metallic rectangle)

Bag casts shadow on pants beneath

Final Polish:

Clean professional appearance

Confident, trustworthy character

Suitable for business, corporate environment

All elements harmonious and coordinated

📋 EXAMPLE 2: HORROR YOUNG ADULT FEMALE (Moderate Intensity)

USER INPUT:

CHARACTER TYPE: Horror AGE: Young Adult (25 years) GENDER: Female SKIN TONE: Pale grey-blue (corpse-like but alive) HAIR: Long black hair, unkempt, hangs heavy EYES: Hollow dark brown, no light reflection FACIAL FEATURES: Slight facial asymmetry, dark under-eye circles EXPRESSION: Emotionless, blank stare CLOTHING: - Dark grey long-sleeve t-shirt, worn appearance - Black jeans, slightly baggy - Dark brown worn boots ACCESSORIES: None SPECIAL NOTES: Psychological horror, unsettling presence, moderate intensity 

EXECUTION BREAKDOWN:

Proportions:

Height: 7.5 head units (standard female)

Shoulders: 2 head widths (narrow feminine)

Waist: 1.6 head widths (slight curve)

Hips: 2.3 head widths (hourglass but subtle)

Posture: Perfectly straight, rigid stillness

Horror Skin Tone Application:

Base: #8BA5B8 (pale blue-grey, sickly)

Shadow: #5B7588 (deep navy blue-grey, 45% darker)

Shadow coverage: 50% of face (heavy, dramatic)

Highlight: MINIMAL - only #9BB5C8 on forehead center (5% lighter, barely visible)

Facial Asymmetry (Subtle):

Left eye 3% larger than right eye (slight difference)

Left eye positioned 1mm higher than right

Mouth corner left side 1mm higher than right

Overall: Not obviously asymmetric, but subconsciously wrong

Facial Details:

Eyes (Horror-Specific):

Iris: Dark murky brown #4A3020 (desaturated, dead)

Pupils: Dilated, 55% of iris size (larger than normal)

Sclera: Light grey #D8D8D8 (not white, lifeless)

NO catch light (CRITICAL - dead stare)

Gaze: Direct at viewer, unblinking, frozen

Under-eye circles: HEAVY purple-brown #6B4A58, extends 7mm below eye, crescent shape

Other Features:

Eyebrows: Thin, black, natural arch (female style)

Nose: Minimal construction, slightly discolored at tip (faint red undertone)

Lips: Dark purple-brown #6B4A58, thin, slightly parted 2mm

Expression: EMOTIONLESS, blank, vacant stare

Horror Skin Details:

Vascular Visibility:

Thin blue-purple veins visible at temples: #6B5B8B (0.8px lines, 50% opacity)

Subtle vein branching pattern, organic curves

3-4 visible veins each temple

Uneven Skin Tone:

Darker patches on left cheek (irregular shape, 1cm, subtle)

Slight grey-green cast around mouth area

Mottled appearance (not uniform color across face)

More yellow-grey around eyes

Discoloration:

Under nose: Faint grey-brown

Around mouth: Slightly darker, dirty appearance

Jawline: Uneven shadow beyond standard shading

Hair Rendering:

Horror Hair Style:

Length: Long, extends to mid-back

Base color: #0A0A0A (dull black, lifeless)

Shadow: #000000 (pure black in depths)

Highlight: MINIMAL - only 2-3 subtle steel-blue strokes #3A5B6C (very faint)

Texture: Heavy, unkempt, weighed down

Construction: 15-18 large chunky segments, irregular

Some strands partially covering left side of face (crosses forehead slightly)

Clumping visible (strands stick together unnaturally)

NOT flowing gracefully - hangs limp and heavy

Clothing Construction:

Dark Grey T-Shirt:

Base: #4A4A4A (dark grey, desaturated)

Shadow: #2A2A2A (charcoal, heavy shadows)

Style: Long sleeves, crew neck, loose fit

Wrinkles: 6-8 fold lines (more than normal, worn appearance)

Worn indicators: 

Slight compression marks at elbows (darker patches)

Fabric appears heavy, hangs on body

Small ambiguous stain on right shoulder: Murky brown #3A2A1A, irregular shape (8mm), 60% opacity

Black Jeans:

Base: #1A1A1A (faded black)

Shadow: #000000 (pure black shadows)

Fit: Slightly baggy, not form-fitting

Wrinkles: Multiple fold lines at knees (7-8 lines)

Worn appearance, not crisp

Hem slightly uneven at ankles

Dark Brown Boots:

Base: #3A2410 (dark worn brown)

Shadow: #2A1A08 (nearly black)

Style: Ankle height, lace-up

Worn leather appearance (subtle texture lines)

Scuffed (slightly lighter patches on toe area)

Minimal shine (matte, old leather)

Horror Lighting (Dramatic):

Light source: Top-left, harsh single-direction light

Deep shadows: 50-55% of character in shadow

Face shadows: Deep in eye sockets, under cheekbones (skull-like effect)

Neck shadows: Heavy on right side, creates hollow appearance

Body shadows: Dramatic contrast, very dark on right side

Horror Atmosphere:

Character appears STILL, frozen, not breathing implied

Aware presence (feels conscious of viewer)

Posture unnaturally straight and rigid

Hands visible at sides, fingers slightly curled, relaxed but wrong

Overall: UNSETTLING not SHOCKING, psychological not gore

Quality Check - Horror Verification: ✅ Desaturated grey-blue skin ✅ No catch light in eyes ✅ Emotionless expression
✅ Facial asymmetry present ✅ Heavy under-eye circles ✅ Skin mottling visible
✅ Vein visibility ✅ Unkempt heavy hair ✅ Dark worn clothing
✅ Ambiguous stain present ✅ Dramatic shadows ✅ Rigid posture
✅ NO gore or obvious monsters ✅ Psychological horror maintained

📋 EXAMPLE 3: NORMAL ELDERLY MALE (Traditional Indian Wear)

USER INPUT:

CHARACTER TYPE: Normal AGE: Elderly (68 years) GENDER: Male SKIN TONE: Medium-tan Indian, with yellow undertones (elderly) HAIR: Grey-white, thinning EYES: Dark brown FACIAL FEATURES: Wrinkles, glasses (rectangular frames), grey mustache EXPRESSION: Gentle smile, kind CLOTHING: - Cream-colored kurta (traditional Indian tunic) - White dhoti (traditional wrap pants) - Brown leather sandals ACCESSORIES: Wooden walking stick in right hand SPECIAL NOTES: Wise, grandfather figure 

EXECUTION BREAKDOWN:

Proportions (Elderly Adjustments):

Height: 6.8 head units (slightly shorter, age-related)

Slight stoop in posture (shoulders forward 5-10°, not extreme)

Thinner build overall

Shoulders: 2.3 head widths (narrower, age-related muscle loss)

Elderly Skin Tone:

Base: #C89060 (medium tan with yellow-grey undertone, less vibrant)

Shadow: #986838 (deep tan-brown, elderly skin darker shadows)

Highlight: #D8A878 (muted light brown)

Skin appears: Muted, less vibrant than young adult, yellow cast

Texture: Slightly mottled (subtle color variation, not perfectly smooth)

Elderly Facial Features:

Wrinkle Lines (MANDATORY):

Forehead: 4 horizontal lines across, 1px weight, curved

Crow's Feet: 4-5 radiating lines from outer corner of each eye, 1px

Nasolabial Folds: Deep curved lines from sides of nose to mouth corners, 1.5px, prominent

Under Eyes: Sagging bags (curved shadow shapes, not just lines)

Above Upper Lip: 3-4 vertical lines, 0.8px, subtle

Cheek Lines: 1-2 diagonal lines from outer eye area toward chin

All lines: Thin, delicate, natural placement

Face Shape:

Sagging jowls (lower face rounder, less defined jaw)

Thinner lips than young adult

Larger nose (cartilage continues growing with age)

Larger ears (also continue growing)

More pronounced nose bridge

Eyes:

Smaller appearing (drooping eyelids)

Dark brown iris #5D3A1A with catch light (normal character)

More pronounced under-eye bags

Upper eyelids heavier, drooping slightly

Grey Mustache:

Color: #C8C8C8 (light grey) base, #9B9B9B (medium grey) shadow

Style: Full mustache, covers upper lip

Texture: Individual stroke marks visible (8-10 strokes)

Slightly bushy, natural (not perfectly groomed)

Soft edges (feathered perimeter with short line segments)

Glasses:

Rectangular black frames: #1A1A1A, 3px thickness

Large lenses (elderly often need larger glasses for reading)

Bridge: 2px connector across nose

Temple arms: Visible extending to ears

Lenses: COMPLETELY TRANSPARENT (no glass visible, see eyes clearly through)

Slight shadow cast on cheeks below frames (subtle, 10% opacity)

Hair Rendering:

Grey-White Hair:

Base: #D8D8D8 (light grey)

Shadow: #A8A8A8 (medium grey)

Highlight: #F0F0F0 (nearly white on top surface)

Style: Thinning, receding at temples

Construction: 6-8 wispy segments (fewer than young adult, thinner)

Visible scalp in some areas (hair less dense)

No dramatic styling, natural elderly hair

Traditional Indian Clothing:

Cream Kurta (Tunic):

Base: #F0E8D0 (cream/off-white)

Shadow: #D0C8B0 (beige shadow)

Style: Long tunic extending to mid-thigh

Mandarin collar (stand-up collar, 2cm tall, 2px outline)

Front closure: Button placket down center-right (6-7 small buttons)

Long sleeves to wrists

Loose, comfortable fit (not tight)

Side slits: Small openings at hem sides (15cm up from bottom)

Fabric folds: Vertical draping folds from shoulders (4-5 major fold lines)

Subtle embroidery at neckline (optional): Thin decorative lines 0.5px, simple pattern

White Dhoti (Wrap Pants):

Base: #FAFAFA (bright white)

Shadow: #E0E0E0 (light grey shadow)

Style: Traditional wrapped lower garment

Appears like loose pants from front view

Gathered at waist (many small vertical folds, 15-20 lines, 1px each)

Loose through legs, gathered at ankles

Traditional draping implied (simplified rendering for front view)

Ends: Fabric hem visible at ankles

Brown Leather Sandals:

Base: #6B4A28 (brown leather)

Shadow: #4A3018 (dark brown)

Style: Simple open-toe sandals, flat

Straps: 2-3 straps visible across foot top

Sole: Thin, flat (no heel)

Toes partially visible (simplified, not detailed)

Accessories:

Wooden Walking Stick (Right Hand):

Position: Held in right hand, standing next to body

Stick: Vertical, extends from ground to hand height (90cm visible)

Color: #6B4A28 (wooden brown) base, #4A3018 (dark brown) shadow

Width: 2cm diameter (2px outline)

Texture: Subtle wood grain (2-3 horizontal lines, 0.5px, suggesting natural wood)

Top handle: Curved or rounded (simplified)

Shadow cast: Thin shadow on ground to right of stick

Expression:

Gentle smile: Mouth corners slightly upturned (3mm)

Eyes: Warm, kind appearance (slight crinkle at corners from smile)

Eyebrows: Relaxed, natural position

Overall: Wise, grandfatherly, approachable demeanor

Posture:

Slight forward lean (elderly stoop, but not extreme)

Weight partially on walking stick (implied)

Relaxed, comfortable stance

Feet closer together than young adult (stable stance)

Cultural Accuracy:

Traditional Indian attire rendered respectfully

Colors appropriate (cream and white common for elderly men)

Style authentic to North Indian traditional dress

Simple, dignified appearance

📋 EXAMPLE 4: NORMAL CHILD FEMALE (Casual, Playful)

USER INPUT:

CHARACTER TYPE: Normal AGE: Child (10 years old) GENDER: Female SKIN TONE: Light peachy (fair) HAIR: Brown, shoulder-length, slightly messy EYES: Large brown eyes FACIAL FEATURES: Freckles across nose and cheeks EXPRESSION: Gentle smile CLOTHING: - Orange t-shirt with simple graphic - Blue denim shorts - White sneakers with pink accents ACCESSORIES: Small yellow backpack SPECIAL NOTES: Cheerful, energetic child 

EXECUTION BREAKDOWN:

Child Proportions (CRITICAL DIFFERENCES):

Height: 6.3 head units (shorter, more compact than adult)

Head: Proportionally LARGER (30% bigger relative to body)

Face: Rounder, fuller cheeks, less defined jaw

Eyes: LARGER (35% bigger than adult proportionally)

Body: Less muscular definition, rounder limbs

Legs: Slightly shorter relative to torso than adult

Child Skin Tone:

Base: #FFD4B0 (light peachy, vibrant, healthy)

Shadow: #F0C8A0 (only 15% darker - children have softer shadows)

Highlight: #FFE8D0 (bright, youthful glow on cheeks, forehead)

Skin: Smooth, even tone, very clean (no blemishes, veins, texture)

Child Facial Features:

Face Shape:

Round oval (not angular)

Fuller cheeks (apple cheeks, youthful plumpness)

Small, soft chin (not pointed or defined)

Barely visible jawline (rounded, baby fat)

Large Eyes:

Size: 40% larger than adult proportionally

Iris: Warm brown #6B4A28

Pupils: Standard size

Sclera: Pure white #FFFFFF (child eyes very clear)

Catch light: Prominent white dot (bright, lively)

Eyelashes: 4-5 delicate strokes upper lid (feminine child)

Expression: Wide, bright, cheerful

Eyebrows:

Thin, delicate (2px height)

Light brown #6B4A28

Natural, unplugged shape

Soft arch

Nose:

Very small, button nose

Minimal line detail (just nostril curves)

Slight upturned tip (cute, childlike)

Mouth:

Small, gentle smile

Thin lips (children have smaller features)

Soft pink #E8B4B4

Corners upturned 4mm

Freckles:

Small dots across nose bridge and upper cheeks

Color: #D4A878 (light tan, subtle)

Size: 1-2px diameter each

Quantity: 12-18 freckles scattered naturally

Placement: Concentrated on nose, spreading to cheeks

Random spacing (not uniform pattern)

Hair Rendering:

Brown Shoulder-Length Hair:

Base: #6B4A28 (medium warm brown)

Shadow: #4A3018 (darker brown)

Highlight: #8B6A48 (light brown streaks)

Length: Extends just past shoulders

Style: Slightly messy, playful (not perfectly styled)

Construction: 12-15 chunky segments, flowing

Some strands out of place (casual, natural)

Movement: Appears slightly windblown (energetic feel)

Clothing Construction:

Orange T-Shirt:

Base: #E86A28 (bright vibrant orange)

Shadow: #C84A08 (darker orange)

Style: Short sleeves, crew neck, casual fit

Simple graphic: Abstract shape or small icon on chest (simplified, 3-4 shapes, not detailed logo)

Colors in graphic: Yellow #F0C848 and white #FFFFFF

Fit: Slightly loose, comfortable (not tight on child)

Wrinkles: Minimal (2-3 lines at armpits only)

Blue Denim Shorts:

Base: #5B7B98 (medium blue denim)

Shadow: #3B5B78 (dark blue)

Style: Above knee length, casual fit

Details: 

White stitching: Double parallel lines 0.8px, gold-tan color

Pockets: Two front pockets, simple outline

Hem: Rolled cuff at legs (small fold, 1.5cm)

Fit: Comfortable, not tight

White Sneakers with Pink Accents:

Base: #F5F5F5 (bright white)

Shadow: #D8D8D8 (light grey)

Style: Athletic sneakers, lace-up

Pink accents: 

Swoosh or stripe on side: #E86A8A (bright pink)

Sole trim: Pink accent line

Laces: White, zigzag pattern (5-6 crossings)

Sole: Thick rubber (2-3cm), white with pink bottom edge

Clean, new appearance (child shoes often well-maintained)

Accessories:

Yellow Backpack:

Position: On back, straps over shoulders

Visible from front: Only two shoulder straps visible

Strap details: 

Yellow fabric: #F0C848 (bright yellow)

Width: 2cm each strap

Padded appearance (slightly puffy)

Over shoulders, parallel

Bag body not visible (behind character)

Straps cast subtle shadow on orange t-shirt

Child-Specific Details:

✅ Simpler features: Less facial detail than adult
✅ Brighter colors: Vibrant, cheerful palette
✅ Softer shading: Only 15-20% darker (vs 25-30% adult)
✅ Minimal wrinkles: Only 1-2 lines at major joints
✅ Clean appearance: No stains, dirt, wear (well-cared-for child)
✅ Proportionally larger head and eyes
✅ Rounder body forms: Less angular than adult

Expression & Mood:

Cheerful, happy, energetic feeling

Innocent, youthful appearance

Appropriate for 10-year-old child

NOT overly cute/chibi style - realistic proportions within style

📋 EXAMPLE 5: HORROR ELDERLY MALE (High Intensity)

USER INPUT:

CHARACTER TYPE: Horror (HIGH INTENSITY) AGE: Elderly (75 years) GENDER: Male SKIN TONE: Pale grey-green, sickly HAIR: Thin white hair, sparse EYES: Sunken, hollow, pale grey eyes FACIAL FEATURES: Extreme wrinkles, gaunt, skeletal appearance EXPRESSION: Slight disturbing smile CLOTHING: - Tattered dark brown coat - Stained grey shirt underneath - Old black pants - Worn boots ACCESSORIES: None SPECIAL NOTES: Extreme psychological horror, unsettling, skeletal but alive 

EXECUTION BREAKDOWN:

Elderly Horror Proportions:

Height: 6.5 head units (shorter, pronounced stoop)

Extreme thinness (skeletal, emaciated appearance)

Shoulders hunched forward significantly (20-30° forward lean)

Neck extended forward (head projects ahead of shoulders)

Arms thin, bony appearance

Legs thin, weak-looking stance

Extreme Horror Skin Tone:

Base: #9BAA98 (pale grey-green, corpse-like but alive)

Shadow: #6B7A68 (deep olive-grey, 45% darker, very dramatic)

Highlight: BARELY PRESENT - only #ABB8A8 (5% lighter, almost imperceptible)

Shadow coverage: 60% of face and body (extreme dramatic lighting)

Elderly + Horror Combination:

Extreme Wrinkle Lines:

Forehead: 6-7 deep horizontal lines (1.5px, very prominent)

Crow's Feet: 7-8 radiating lines, deep and long

Nasolabial Folds: VERY DEEP (2px lines, shadow-filled valleys)

Marionette Lines: From mouth corners down to chin (2px)

Vertical Lip Lines: 5-6 lines above upper lip

Under Eyes: Extreme sagging, deep bags

Cheeks: Sunken, hollow (heavy shadow creates skeletal appearance)

Neck: Multiple horizontal lines, loose skin appearance

Gaunt, Skeletal Face:

Cheekbones VERY prominent (sharp edges, deep shadows beneath)

Temples sunken (concave areas, dark shadows)

Eye sockets DEEP (eyes appear recessed 5mm into skull)

Jaw angular and pronounced (skin tight over bone)

Chin sharp and pointed

Skull-like appearance (but still recognizable as living human)

Horror Facial Features:

Sunken Hollow Eyes:

Position: Deep in eye sockets

Iris: Pale murky grey #9BA8A8 (nearly colorless, faded)

Pupils: Dilated, 60% of iris (too large)

Sclera: Yellowed grey #D8D4C8 (aged, unhealthy)

NO catch light (dead, lifeless stare)

Bloodshot: 3-4 thin red veins visible #8B3030 (0.5px lines)

Under eyes: EXTREME dark circles, purple-black #4A3A48, extends 12mm below

Eyelids: Heavy, drooping, almost closing

Disturbing Slight Smile:

Mouth: Corners upturned 2mm (barely visible smile)

Asymmetric: Left corner 1mm higher than right

Smile doesn't reach eyes (eyes remain dead)

Thin lips: Dark purple-grey #6B5B68

Teeth: If visible through parted lips, yellowed #E8E0C8, some gaps

Extreme Skin Horror Details:

Vascular Visibility (Prominent):

Blue-purple veins at temples: #6B5B8B (1px lines, 70% opacity, VERY visible)

Branching patterns across forehead

Veins visible on hands (back of hands)

Network of vessels, prominent

Severe Discoloration:

Around eyes: Deep purple-brown bruising appearance

Mouth area: Grey-brown, dirty look

Jawline: Uneven grey-green patches

Overall: Heavily mottled skin (not uniform at all)

Texture:

Dry, papery skin (subtle texture marks, 0.3px scattered)

Liver spots: Small dark brown spots #5B3A28 (2-3mm, scattered on face and hands)

Cracked lips: Thin crack lines (0.5px)

Hair:

Sparse Thin White Hair:

Base: #E8E8E8 (dull white-grey)

Shadow: #B8B8B8 (medium grey)

Style: VERY thin, balding

Visible scalp: Pale grey-green skin showing through

Construction: Only 4-6 wispy segments (minimal hair)

Texture: Fine, flyaway, unkempt

No volume, lays flat on scalp

Extreme Horror Clothing:

Tattered Dark Brown Coat:

Base: #3A2410 (very dark brown, dirty)

Shadow: #1A1008 (nearly black)

Style: Long coat to mid-thigh, worn and aged

Condition: 

Frayed edges: Irregular hem, torn appearance (small notches, 0.5px texture)

Holes: 1-2 small holes visible (irregular dark spots)

Heavily worn: Shiny/compressed patches at elbows, shoulders

Appears heavy, hangs loosely on thin frame

Multiple stains: 

Dark brown stain on right shoulder (3cm, irregular): #2A1408

Grey-brown stain on left chest (2cm): #4A3A2A

Dark reddish-brown stain on sleeve (ambiguous, 1.5cm): #4A1A1A

All stains 60-70% opacity, mysterious origin

Many wrinkles: 10-12 fold lines (fabric worn continuously)

Stained Grey Shirt (Underneath, Partially Visible):

Base: #6B6B6B (dark grey)

Visible at neckline and through coat gaps

Heavily stained: Multiple dark patches

Collar askew, twisted

Old Black Pants:

Base: #1A1A1A (faded black, worn)

Shadow: #000000

Baggy on thin frame (hangs loosely)

Faded, worn appearance

Multiple wrinkles and creases

Worn Boots:

Base: #2A1A0A (very dark brown, nearly black)

Cracked leather appearance

Scuffed, damaged

Laces broken or missing (simplified)

Extreme Horror Lighting:

Single harsh light source: Top-left

60% of character in deep shadow

Face: Skull-like shadows under cheekbones, eye sockets appear black

Body: Deep contrast, very dramatic

Creates maximum unsettling effect

**High Intensity Horror Elements

:**

✅ Extreme grey-green skin
✅ Skeletal, gaunt appearance
✅ Severe wrinkles and age indicators
✅ Deep sunken eyes, no catch light
✅ Disturbing asymmetric smile
✅ Prominent vascular visibility
✅ Severe skin discoloration and mottling
✅ Sparse, thin hair revealing scalp
✅ Tattered, stained clothing (multiple ambiguous stains)
✅ Extreme dramatic shadows (60% coverage)
✅ Liver spots and texture
✅ Emaciated, skeletal but STILL HUMAN

CRITICAL: Still NO gore, no open wounds, no obvious monster

Character is DISTURBING not GROTESQUE

Psychological horror through wrongness

Appears alive but shouldn't be

Maximum unsettling within human framework

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 9 OF 10

🎓 ADVANCED TECHNIQUES, TROUBLESHOOTING & EXPERT REFINEMENTS

🎯 ADVANCED CHARACTER DIFFERENTIATION:

ACHIEVING UNIQUE CHARACTERS WITHIN STYLE CONSISTENCY:

Problem: Multiple characters in same style can look too similar

Solution - Differentiation Vectors:

1. FACIAL STRUCTURE VARIATION (Within Proportional Rules):

Face Shapes:

Oval: Standard, balanced (most common)

Round: Fuller cheeks, softer jawline, wider proportions

Square: Angular jaw, strong chin, defined edges

Heart: Wider forehead, narrower pointed chin

Diamond: Narrow forehead and jaw, wide cheekbones

Long/Rectangular: Elongated proportions, longer nose-to-chin

Feature Placement Micro-Adjustments:

Eye spacing: Standard (1 eye width) vs Slightly Wide (1.2 eye width) vs Close-Set (0.8 eye width)

Eye slant: Horizontal vs Slight upturn (almond) vs Slight downturn (droopy)

Nose size: Standard vs Large/Prominent vs Small/Button

Nose bridge: Straight vs Slight curve vs Strong bump

Mouth width: Standard (aligns with iris centers) vs Wide vs Narrow

Lip fullness: Thin vs Medium vs Full

Chin: Standard vs Prominent vs Recessed vs Pointed

Implementation:

Character A: Oval face, standard spacing, straight nose, medium lips Character B: Round face, wide-set eyes, button nose, full lips Character C: Square face, close-set eyes, prominent nose, thin lips 

Result: Three distinctly different faces within same style system

2. BUILD & BODY TYPE VARIATION:

Male Body Types:

Slim/Lean: Shoulders 2.3 heads, minimal muscle definition, thin limbs

Average: Shoulders 2.6 heads, subtle muscle, standard proportions

Athletic/Muscular: Shoulders 3 heads, defined pecs/arms, broader chest

Heavy/Stocky: Shoulders 2.8 heads, thicker torso, less waist taper, rounder

Tall/Lanky: 8.5 head heights, thin limbs, elongated proportions

Female Body Types:

Petite: 6.8 head heights, delicate frame, narrow shoulders (1.8 heads)

Average: Standard 7-7.5 heads, balanced proportions

Athletic: Defined muscles visible, broader shoulders (2.2 heads), less curve

Curvy: Pronounced hourglass, wider hips (2.8 heads), fuller bust

Plus-Size: Rounder forms, fuller torso, less defined waist curve, thicker limbs

Age Variations:

Late Teens (16-19): Slightly smaller frame, less filled-out

Young Adult (20-35): Standard proportions, peak physical condition

Middle Age (36-55): Slight thickening, less defined muscles, minor settling

Elderly (60+): Thinner or heavier extremes, muscle loss, stooped posture

3. EXPRESSION RANGE (Normal Characters):

Neutral Expressions:

Relaxed Neutral: Mouth closed flat line, eyes open standard

Serious Neutral: Slight furrow between brows, firm mouth line

Tired Neutral: Heavy eyelids, slight downturn of features

Confident Neutral: Slight eyebrow raise, direct gaze

Positive Expressions:

Subtle Smile: Mouth corners up 2-3mm, no teeth, eyes unchanged

Genuine Smile: Mouth corners up 5mm, slight eye crinkle (crow's feet begin)

Bright Smile: Teeth visible (upper row), eyes narrow slightly, cheek raise

Gentle/Kind: Soft smile, relaxed eyebrows, warm eye contact

Negative Expressions:

Concerned: Eyebrows drawn together and down, slight frown

Sad: Mouth corners down, eyebrows inner corners raised, eyes downcast

Frustrated: One eyebrow raised, mouth line firm or twisted

Tired/Weary: Heavy eyelids, slight frown, overall drooping features

Professional/Formal:

Polite Smile: Small controlled smile, eyes neutral (service industry)

Serious Professional: Neutral with slightly raised eyebrows (attentive)

Authoritative: Firm mouth, direct intense gaze, strong eyebrows

Implementation Notes:

Keep expressions SUBTLE for realism

Avoid extreme cartoon expressions

Maintain professional illustration quality

Expression should match character context

CLOTHING PERSONALITY INDICATORS:

Color Psychology in Wardrobe:

Professional/Serious:

Navy blue, charcoal grey, black, white

Conservative, traditional combinations

Minimal patterns, solid colors dominant

Creative/Artistic:

Unusual color combinations (teal + orange, purple + yellow)

Patterns: Geometric, artistic prints

Layered, eclectic styling

Approachable/Friendly:

Warm colors: Soft blues, greens, earth tones

Comfortable casual styles

Rounded shapes in clothing design

Bold/Confident:

Bright reds, bold blacks, striking contrasts

Sharp, tailored fits

Statement pieces

Subdued/Introverted:

Muted tones, greys, soft pastels

Loose, comfortable fits

Minimal accessories

🔧 TROUBLESHOOTING COMMON ISSUES:

ISSUE 1: Character Looks "Flat" or "Lifeless"

Diagnosis: Insufficient shading, no depth, poor contrast

Solutions:

✅ Increase Shadow Darkness:

Shadows may be too light (only 15-20% darker)

Increase to 30-35% darker for normal, 40-50% for horror

Check: Place character against white background - shadows should be clearly visible

✅ Expand Shadow Coverage:

Shadows covering too little (10-20% only)

Increase to 35-45% coverage for proper depth

Major forms need shadow definition

✅ Add Secondary Shadows:

Deep recesses need extra shadow layer

Armpits, under collar, behind knees = darkest areas

3-tone shading in complex areas

✅ Verify Light Direction Consistency:

All shadows must point same direction

Check every shadow matches light source angle

Inconsistent lighting = flat appearance

ISSUE 2: Character Looks "Too Cartoony" or "Childish"

Diagnosis: Over-simplified features, wrong proportions, too cute

Solutions:

✅ Increase Anatomical Accuracy:

Check proportions against head-height system

Adult characters: 7.5-8 heads (not 6-7)

Eyes not too large (common mistake)

Features properly spaced

✅ Add Subtle Detail:

Nose needs more definition (not just two dots)

Ears need interior structure (not just C-shape)

Hands need finger separation (not mittens)

Clothing needs realistic folds

✅ Refine Line Weight:

Outlines may be too thick (4-5px = cartoony)

Reduce to 2-3px for professional look

Interior lines 1-1.5px maximum

✅ Mature Color Palette:

Avoid oversaturated primary colors

Use muted, professional tones

Add more shadow complexity

✅ Expression Control:

Avoid exaggerated expressions

Keep features subtle and restrained

Professional neutrality

ISSUE 3: Gender Ambiguous or Incorrect

Diagnosis: Body proportions not differentiated, features too similar

Solutions:

✅ Verify Shoulder-Hip Ratio:

MALE: Shoulders WIDER than hips (2.6:2.0 ratio)

FEMALE: Hips WIDER than shoulders (2.0:2.4 ratio)

This is CRITICAL and non-negotiable

✅ Chest Structure (MANDATORY):

MALE: Flat chest with subtle pec definition

FEMALE: Bust MUST be present and anatomically rendered 

Curved line suggesting breast shape

Shadow under bust for dimension

Proportionate to body frame

NEVER flat-chested (unless child)

✅ Facial Features:

MALE: Angular jaw, square chin, thicker eyebrows (4-6px), minimal lashes

FEMALE: Softer jaw, rounded/pointed chin, thin eyebrows (2-3px), prominent lashes (4-6 strokes)

✅ Neck & Hands:

MALE: Thicker neck (1 head width), larger hands

FEMALE: Slender neck (0.7 head width), smaller delicate hands

✅ Clothing Fit:

Even if wearing unisex clothing, fit differs

Male clothing hangs straighter

Female clothing follows curves

ISSUE 4: Horror Character Too Obvious or "Monster-Like"

Diagnosis: Over-designed horror elements, lost subtlety

Solutions:

✅ Reduce Extreme Elements:

Remove any fangs, claws, inhuman features

Eliminate excessive blood/gore

Tone down supernatural indicators

Keep within human framework

✅ Increase Subtlety:

Facial asymmetry should be BARELY noticeable (3-5% difference)

Skin discoloration subtle patches (not half green/half grey)

Stains small and ambiguous (not obviously blood)

Wrongness through implication, not statement

✅ Maintain Human Proportions:

Horror character still has normal anatomy

7-8 head heights maintained

No elongated limbs, oversized features

Recognizable as human at first glance

✅ Psychological vs Visual Horror:

Fear from EXPRESSION and PRESENCE, not appearance

Emotionless stare more disturbing than screaming face

Stillness more unsettling than action

Awareness more creepy than aggression

ISSUE 5: Clothing Looks "Painted On" or Unnatural

Diagnosis: No folds, doesn't follow body, poor draping

Solutions:

✅ Add Appropriate Fold Lines:

Minimum 3-5 folds per major joint area

Compression folds at elbows, knees, armpits

Hanging folds on loose garments

Reference Part 5 fold system

✅ Show Fabric Weight:

Light fabrics: Many small wrinkles

Heavy fabrics: Few large folds

Different garments behave differently

✅ Layer Correctly:

Inner garments visible at edges

Overlap shows depth

Collar/cuffs/hems defined

✅ Proper Fit:

Tight clothing follows body curves closely

Loose clothing hangs from high points

Realistic tension and compression

ISSUE 6: Accessories Look "Floating" or Disconnected

Diagnosis: Poor integration, wrong positioning, no physics

Solutions:

✅ Attach Properly:

Jewelry rests ON skin (small indent/pressure point)

Bags have straps that follow body curves

Glasses rest on nose bridge and ears

Everything connects logically

✅ Show Weight:

Heavy bag pulls strap tight

Watch band wraps around wrist completely

Necklaces hang with gravity

Equipment has realistic positioning

✅ Cast Shadows:

Accessories cast shadows on body beneath

Glasses shadow on cheeks

Necklace shadow on chest

Bag shadow on hip/leg

✅ Interaction Marks:

Watch compresses wrist skin slightly

Belt tightens fabric around waist

Shoulder bag strap indents shirt slightly

🎨 EXPERT REFINEMENT TECHNIQUES:

MICRO-DETAILS FOR ELITE QUALITY:

Skin Realism (Normal Characters):

✅ Subtle Blush (Optional):

Very faint pink/red on cheeks (#E8B4B4, 20% opacity)

Circular soft area (2-3cm diameter)

Suggests healthy blood flow

Appropriate for young females, children, cold environments

✅ Forehead Shine (Selective):

Small highlight on center forehead (5-10% lighter)

Suggests natural skin oils

Appropriate for warm environments, male characters

Keep very subtle

✅ Five O'Clock Shadow (Male):

Tiny dots along jawline, chin, upper lip

Very subtle, 30% opacity grey-brown

Density variation (heavier on chin, lighter on cheeks)

0.5px marks, scattered naturally

✅ Lip Detail Enhancement:

Cupid's bow emphasized (M-shape upper lip)

Center of lower lip slightly lighter (highlight)

Corners slightly darker (shadow)

Vertical lip texture lines (very subtle, 0.3px, 2-3 lines)

Hair Realism:

✅ Flyaway Strands:

2-4 single thin lines (0.8px) breaking from main hair mass

Adds natural, lived-in appearance

Positions: Temples, crown, around ears

Don't overdo (maintain clean look)

✅ Hair Part Line:

Visible scalp line where hair parts (if applicable)

Thin line (0.5px) showing skin color

Natural zigzag (not perfectly straight)

2-4cm visible portion

✅ Hairline Detail:

Small notches and irregularities at hairline

Not perfectly smooth edge

Baby hairs (tiny wisps at forehead, 0.5px, 1-3 strands)

Natural recession at temples (especially males)

✅ Hair Volume Depth:

Front chunks: Lighter color (catching more light)

Middle chunks: Base color

Back chunks: Darker (in shadow)

Creates three-dimensional illusion

Clothing Realism:

✅ Seam Stitching:

Double parallel lines along major seams (0.8px each, 1-2mm apart)

Contrasting thread color (gold on blue denim, white on black fabric)

Visible on shoulder seams, side seams, inseams

Professional craftsmanship indicator

✅ Fabric Sheen (Selective):

Smooth fabrics (silk, satin, leather): Thin highlight streaks

Position: Along curves and high points

30-50% opacity white

Directional (follows fabric flow)

✅ Button Holes:

Small horizontal line next to each button (3-4mm long, 0.5px)

Shows button actually functions

Same color as contrasting stitching

Only on visible buttons (front of shirts, cuffs)

✅ Pocket Realism:

Slight shadow inside pocket opening (suggests depth)

Thread color stitching around perimeter

Functional appearance (not just drawn on)

Slight bulge if items inside (optional)

✅ Hem Detail:

Double-fold hem visible at cuffs, collar, bottom edges

Two parallel lines (0.5-1px) showing folded construction

Stitching line along hem

Professional garment construction

Eye Realism (Advanced):

✅ Iris Detail (Optional - Use Sparingly):

4-6 thin radial lines from pupil outward (0.3px, 30% opacity)

Suggests iris muscle structure

Same color as iris, just slightly darker

Keep very subtle (overdone = looks weird)

✅ Limbal Ring:

Thin dark outline around iris perimeter (0.8px)

Darker than iris color

Creates eye definition and "pop"

Common in young healthy eyes

✅ Waterline:

Thin pink-red line along inner lower eyelid (0.5px, subtle)

Color: #E8A4A4

Shows eye moisture and realism

Very subtle, barely visible

✅ Sclera Detail:

Not pure white - slight off-white or cream (#F8F8F8)

Very subtle grey shadow in corners

Bloodshot (if character tired): 1-2 thin red vessels (0.3px, 20% opacity)

Hand Detail (When Visible and Prominent):

✅ Knuckle Definition:

Small curved lines at knuckle bends (0.8px)

2-3 lines per finger segment

Shows articulation and realism

Only if hands are focal point

✅ Fingernail Detail:

Small curved lines at fingertips (1px)

Lunula (half-moon at nail base) visible (0.5px curve)

Nail color: Pink #E8D4D4 or painted (females)

Subtle shine highlight on nail surface

✅ Palm Lines:

2-3 major lines across palm (0.5px)

Natural placement (life line, heart line, head line)

Very subtle, not prominent

Only if palm facing viewer

✅ Finger Spacing:

Fingers slightly separated (not pressed together)

Natural relaxed curl

Thumb at angle from hand

Realistic joint bends

🎯 ADVANCED HORROR TECHNIQUES:

SUBTLE PSYCHOLOGICAL HORROR ENHANCEMENT:

Micro-Expression Wrongness:

✅ Incomplete Expression:

Smile only in mouth (eyes remain dead)

Eyebrows concerned but mouth neutral

One side of face slightly different expression than other

Suggests internal conflict or wrongness

✅ Timing Wrongness:

Expression seems frozen mid-transition

Like character was interrupted

Not completing natural expression arc

Unsettling incompleteness

✅ Intensity Mismatch:

Very slight smile when should be neutral (too subtle to be friendly)

Almost-frown that suggests suppressed emotion

Expression that doesn't match context

Creates cognitive dissonance in viewer

Advanced Skin Horror:

✅ Subcutaneous Shadows:

Dark areas under skin (not on surface)

Suggests something beneath (without showing what)

Purple-grey zones around eyes, cheeks

Depth that shouldn't exist

✅ Temperature Indicators:

Blue-grey tones suggest cold

Purplish suggests poor circulation

Yellowish suggests jaundice/decay

Color tells physiological story

✅ Moisture Wrongness:

Skin too dry (cracked, papery)

OR too moist (unnatural sheen in wrong places)

Either extreme = unsettling

Normal skin has balanced appearance

Clothing Horror Details:

✅ Temporal Wrongness:

Clothing worn too long (compression beyond normal)

Dust or grime buildup (subtle grey overlay, 10% opacity)

Fabric degradation (small holes, fraying)

Suggests time passage without change

✅ Stain Storytelling:

Position tells story (shoulder = leaning against something)

Color suggests source (brown = dirt/old blood, grey = unknown)

Age of stain (old stains have faded edges, new are darker)

Multiple stains = multiple incidents

Mystery through suggestion

✅ Mismatched Care:

One garment clean, another filthy

Buttons misaligned or wrong holes

Inside-out details visible

Suggests mental disturbance or loss of care

Environmental Interaction (Horror):

✅ Absence of Normal Indicators:

No breath fog (in cold environment)

No sweat (in warm environment)

No natural responses

Wrongness through what's MISSING

✅ Stillness Quality:

No implied movement (hair doesn't suggest recent motion)

Clothing hangs perfectly still

No life indicators

Frozen quality

📊 QUALITY METRICS FOR ELITE CHARACTERS:

Professional Standard Checklist:

TECHNICAL EXCELLENCE: ✅ Lines are vector-clean, no pixelation at 400% zoom
✅ Colors are precisely matched to palette specifications
✅ Shadows have hard edges with zero gradient blur
✅ Proportions mathematically accurate to system
✅ Transparency is complete with clean edges
✅ Resolution meets 3000x4000px at 300 DPI minimum

ARTISTIC QUALITY: ✅ Character has unique, memorable appearance
✅ Style consistency maintained throughout
✅ Details support character narrative
✅ Nothing appears "AI-generated" or generic
✅ Professional artist hand-crafted quality
✅ Suitable for commercial publication

ANATOMICAL ACCURACY: ✅ Gender clearly identifiable from body structure
✅ Age appropriate features and proportions
✅ Realistic human anatomy (within stylized framework)
✅ Clothing drapes naturally on body
✅ Accessories positioned functionally

STYLISTIC CONSISTENCY: ✅ Matches "Kahaani Monday" reference style
✅ 2D flat aesthetic with cel-shading
✅ Clean linework, bold outlines
✅ Proper color saturation for character type
✅ Appropriate detail level (not over or under)

CHARACTER COMMUNICATION: ✅ Personality evident from visual design
✅ Profession/role clear from clothing/accessories
✅ Age and background communicated effectively
✅ Emotional state/expression appropriate
✅ Cultural elements (if any) accurately represented

🔬 EDGE CASES & SPECIAL SITUATIONS:

EDGE CASE 1: Character with Prosthetics/Disabilities

Approach:

Render assistive devices with same style consistency

Wheelchairs, canes, prosthetics = simplified 2D shapes

2px outlines, appropriate materials (metal = grey with shine, plastic = matte)

Integrate naturally (not awkward or focus-stealing)

Respectful, normalized representation

Character is person first, device is just element

EDGE CASE 2: Androgynous or Non-Binary Characters

Approach:

Blend gender indicators subtly

Shoulder-hip ratio: Balanced (2.2:2.2 or 2.3:2.2)

Facial features: Mix of angular and soft

Clothing: Unisex or mixed gender styling

Build: Average, not extreme in either direction

Let character description guide specifics

Respectful, authentic representation

EDGE CASE 3: Fantasy Elements in Realistic Style

If user requests:

Pointed ears (elf): Extended ear tips, still simplified C-shape

Unusual eye colors (purple, red): Rendered same as normal eyes, just different hue

Unusual hair colors (blue, pink): Same rendering technique, different base color

Keep within 2D flat style framework

No magical glows or supernatural effects

Grounded, realistic rendering of fantastic elements

EDGE CASE 4: Historical/Period Clothing

Approach:

Research authentic garment construction

Simplify details to match style level

Maintain period-accurate silhouette and layering

Use appropriate color palettes for era

Don't over-detail (keep style consistent)

Focus on recognizable period indicators

EDGE CASE 5: Heavily Tattooed/Scarred Characters

Tattoos:

Simplified designs (no complex realistic tattoos)

1-1.5px outlines for tattoo linework

Flat color fills (no tattoo shading complexity)

Follow skin contours and body curves

Visible but not overwhelming

Scars:

Thin lines (0.8-1px) in lighter or darker tone than skin

Subtle, not dramatic

Appropriate placement for backstory

Healed appearance (not fresh wounds)

🔥 GOD-TIER ULTRA-PROFESSIONAL 2D CHARACTER CREATION PROMPT

PART 10 OF 10 - FINAL MASTER INTEGRATION

🏆 COMPLETE SYSTEM SUMMARY & QUICK REFERENCE GUIDE

📋 MASTER PROMPT STRUCTURE - CONSOLIDATED:

SECTION A: CORE IDENTITY & CLASSIFICATION

CHARACTER TYPE DETERMINATION (CRITICAL FIRST STEP):

IF USER SPECIFIES "HORROR" → Apply ALL Horror Rules (Part 4) IF USER SPECIFIES "NORMAL" or No Specification → Apply Standard Rules 

MANDATORY CLASSIFICATIONS:

Type: Normal or Horror

Age: Kids (8-14) | Young Adult (18-35) | Middle Age (36-55) | Elderly (60+)

Gender: Male or Female

Ethnicity/Skin Tone: Specific description for accurate color selection

SECTION B: TECHNICAL FOUNDATION

ABSOLUTE REQUIREMENTS (NON-NEGOTIABLE):

✅ View: Perfect front-facing, 0° angle, completely straight, centered
✅ Background: Fully transparent (PNG alpha channel), zero environment
✅ Resolution: Minimum 3000x4000px at 300 DPI
✅ Style: 2D flat illustration, cel-shaded, NO 3D rendering
✅ Linework: Clean vector-quality, 2-3px main outlines, black #000000
✅ Proportions: Accurate to age/gender system (6-8 head heights)
✅ Posture: Standing straight, symmetrical, arms at sides, static pose

SECTION C: ANATOMICAL SYSTEMS

PROPORTIONAL FRAMEWORKS:

ADULT (7.5-8 head heights):

Head: 1 unit

Torso: 2.5-3 units

Legs: 4-4.5 units

Arms reach mid-thigh

GENDER-SPECIFIC BODY STRUCTURE (MANDATORY):

MALE:

Shoulders: 2.5-3 head widths (BROADER)

Waist: 2 head widths (STRAIGHT)

Hips: Equal to waist

Chest: FLAT with subtle pec definition

Neck: 1 head width (THICKER)

Jaw: Angular, square

Eyebrows: Thick (4-6px)

FEMALE:

Shoulders: 2-2.3 head widths (NARROWER)

Waist: 1.5-1.7 head widths (CINCHED)

Hips: 2-2.5 head widths (WIDER - hourglass)

Chest: BUST MUST BE PRESENT (curved line, shadow beneath, anatomically accurate)

Neck: 0.7 head width (SLENDER)

Jaw: Soft, rounded or pointed

Eyebrows: Thin (2-3px), arched

KIDS (6-6.5 head heights):

Larger head proportionally (30% bigger)

Rounder face, fuller cheeks

Larger eyes (35-40% bigger)

Less defined features

Rounder limbs, less angular

ELDERLY (6.5-7 head heights):

Slight stoop, shorter

Wrinkle lines MANDATORY: forehead (3-4), crow's feet (3-5), nasolabial (deep)

Thinner features, sagging skin

Larger nose and ears

Muted skin tones with yellow undertones

SECTION D: COLOR SYSTEMS

NORMAL CHARACTER SKIN TONES:

Light: #FFD4B0 (fair) → #F4C9A0 (light) → #E8C4A0 (light tan)
Medium: #D4A876 (beige) → #C89872 (golden) → #B88860 (medium brown)
Dark: #A87850 (tan brown) → #8B6840 (warm brown) → #6B5030 (rich brown)

Shadow: 25-30% darker than base
Highlight: 10-15% lighter than base
Coverage: Shadows 30-40% of surface

HORROR CHARACTER SKIN TONES:

Grey: #9BA5B0 (pale grey) → #A8B0B8 (ash grey)
Blue: #8BA5B8 (pale blue) → #6B8B9B (steel blue)
Green: #9BAA98 (pale green) → #8B9B8A (grey-green)
Purple: #A89BC7 (pale purple) → #9B8BA8 (mauve)

Shadow: 35-45% darker than base
Highlight: MINIMAL (5-10% lighter, barely visible)
Coverage: Shadows 40-60% of surface (dramatic)

CEL-SHADING TECHNIQUE (CORE SYSTEM):

2-TONE STANDARD:

Base color: Flat fill, entire area

Shadow color: 25-35% darker, hard edge, organic curved shapes

NO GRADIENTS - clean vector edge transitions only

Light source: Top-front-left, 45° angle, ALWAYS consistent

Shadow Placement:

Face: Right side, under cheekbones, jawline, under nose, eye sockets

Body: Right side, armpits, sides, under overlapping layers

Clothing: Folds (valleys), sides, under collars/pockets

Hair: Under chunks, roots, back layers

SECTION E: LINEWORK SPECIFICATIONS

LINE WEIGHT HIERARCHY:

Outer Silhouette: 2.5-3px (black #000000)

Major Divisions: 2-2.5px (body segments, clothing boundaries)

Facial Features: 1.5-2px (eyes, nose, mouth)

Interior Details: 1-1.5px (clothing folds, seams)

Fine Details: 0.8-1px (hair strands, wrinkles, textures)

LINE QUALITY STANDARDS:

✅ Vector-clean smooth curves
✅ Sharp corners where appropriate
✅ Continuous paths, no gaps
✅ Consistent weight along path
✅ Clean intersections (T-junctions precise)
✅ Closed shapes for color fills

SECTION F: FACIAL CONSTRUCTION

STANDARD FACE (NORMAL):

Eyes:

Almond shape, 1.5-2px outline

Iris: Colored circle, 1.5px outline

Pupil: Black circle, centered

Catch light: White dot upper-left (3-5px) - CREATES LIFE

Sclera: Off-white #F5F5F5

Female: 4-6 eyelash strokes (curved, tapering, 1.5px base)

Male: No lashes or minimal

Eyebrows:

Male: Thick solid blocks (4-6px height), straight

Female: Thin arched (2-3px height), refined

Nose:

Minimal: Two curved nostril lines (parentheses)

Optional: Single bridge line (1px)

Shadow triangle beneath for definition

Mouth:

Central line, cupid's bow (M-shape upper lip)

Male: Thin, single tone (#C8857A)

Female: Fuller, two-tone (center lighter, edges darker, #D4A5A5)

Ears:

Simplified C-shape, 2px outer, 1px inner curve

From eyebrow to nose base vertically

HORROR FACE MODIFICATIONS:

✅ Eyes: NO catch light (dead stare), larger pupils (50-60%), grey sclera (#D8D8D8)
✅ Skin: Desaturated grey/blue/green base, mottled/uneven tone
✅ Under-eyes: HEAVY dark circles (purple-brown #6B4A58, 7-10mm below eye)
✅ Asymmetry: One eye 3-5% larger/higher (subtle wrongness)
✅ Expression: Emotionless blank OR frozen mid-breath OR subtle wrong smile
✅ Veins: Visible blue-purple lines at temples (0.5-1px, 40-60% opacity)
✅ Discoloration: Patches around eyes/mouth, mottled appearance

SECTION G: HAIR RENDERING

CONSTRUCTION METHOD:

Chunk-Based System:

8-20 large geometric segments (NOT individual strands)

2-3px outlines per chunk

Interior separation lines (1-1.5px, 2-4 per chunk)

Overlapping layers create depth

Colors:

Black: #1A1A1A base, #000000 shadow, #4A6B7C highlights (steel blue)

Brown: #6B4A28 base, #4A3218 shadow, #8B6A48 highlights

Blonde: #D4B878 base, #B89860 shadow, #F0D8A8 highlights

Grey/White (elderly): #C8C8C8 base, #9B9B9B shadow, #E8E8E8 highlights

Highlight Strokes (CRITICAL):

3-8 curved strokes per major chunk

1-2px width, following hair flow

15-25% brighter than base

On "top" surface of segments

Creates shine and dimension

Horror Hair:

Dull, lifeless color (#0A0A0A)

Heavier, weighed down appearance

Clumping, irregular chunks

Minimal highlights (2-3 only, very subtle)

Unkempt, unwashed look

SECTION H: CLOTHING CONSTRUCTION

FOLD SYSTEM:

Compression Folds (Fabric Pushed):

Location: Bent elbows, knees, armpits, gathered waist

Appearance: 3-6 parallel curved lines (1-1.5px)

Direction: Perpendicular to stress point

Tension Folds (Fabric Pulled):

Location: Shoulder seams, pulled areas

Appearance: Radiating lines from origin (1px)

Direction: All point to tension source

Hanging Folds (Gravity):

Location: Loose skirts, robes, coats

Appearance: 2-4 vertical curved lines (1.5-2px)

Direction: Downward

Fold Shading:

Valley (recess) = shadow color

Peak (raised) = base color

Both sides of fold line get definition

GARMENT TYPES QUICK REFERENCE:

Shirts: Button details (3-4px circles), collar (pointed triangles), seams (1.5px), 4-6 folds at stress points

Pants: Crease line (formal, 1.5px vertical), belt loops (6-8 rectangles), pockets (diagonal line), hem folds at ankles

Dresses: Bust definition (curved line + shadow), waist cinching, hem at knee/thigh, neckline variations

Jackets: Lapels (large triangular folds), buttons (6-8px), pockets (flap style), shoulder structure (padding)

Traditional (Indian): Salwar kameez (loose tunic + gathered pants), dupatta (draped scarf), appropriate colors/embroidery

HORROR CLOTHING:

✅ Dark muted colors: #3A3A3A (charcoal), #2A2A2A (black), #4A3A2A (dark brown)
✅ Worn appearance: 8-12 fold lines (vs 4-6 normal), compression marks
✅ Ambiguous stains: 1-3 small (5-15mm), murky brown-grey or red-brown, 50-70% opacity, mysterious
✅ Heavy fabric: Hangs with weight, not crisp
✅ Aged indicators: Frayed edges, slight holes, faded areas

SECTION I: ACCESSORIES & EQUIPMENT

FOOTWEAR:

Men's Formal: Oxford shoes (laced, toe cap line, shine highlight, #1A1A1A)
Men's Casual: Sneakers (thick sole, panels, laces 6-8 crossings, white or colored)
Women's Formal: Heels (stiletto or block, 7-12cm heel, pointed/rounded toe, shine)
Women's Casual: Flats (no heel, simple, rounded toe, bow detail optional)
Boots: Above ankle, many laces (8-12), thick sole, rugged

JEWELRY:

Earrings: Hoops (circles 8-40mm), studs (3-5px), danglers (hanging elements)
Necklaces: Chain (0.5-1px line), pendants (1-2cm decorative), crosses (religious)
Watches: Band + face (circular 1-1.5cm, glass highlight, metallic)
Rings: Thin band on finger (1px), optional gemstone (2-3px, highlight dot)

Metallic Rendering:

Gold: #D4A858 base, #F0C878 highlight (curved white streak 30-50% opacity)

Silver: #C8C8C8 base, #E8E8E8 highlight

BAGS:

Messenger: Diagonal strap (3cm wide, 2px outline), bag at hip (15-25cm), buckle closure
Backpack: Two shoulder straps visible (2-3cm each), bag behind (not visible front view)
Handbag: Small rectangle (10-15cm), handle or strap, clutch/shoulder/tote styles

PROFESSIONAL EQUIPMENT:

Medical: Stethoscope (black rubber tubing U-shape, metal chest piece, silver)
Police: Badge (shield/star, metallic shine, left chest), duty belt (wide 5-6cm, equipment pouches)
Clergy: Cross necklace, clerical collar (white band 2-3cm at neck)
Worker: Tool belt (pouches, tools protruding), utility vest (many pockets)

SECTION J: WORKFLOW PROCESS

10-STEP EXECUTION:

Analyze: Parse user description → Classify type/age/gender → Plan approach

Sketch: Proportional framework → Pose verification → Volume addition

Linework: Outer silhouette → Major divisions → Facial features → Details

Base Colors: Skin → Hair → Eyes → Clothing → Accessories (flat fills)

Shading: Calculate shadow colors → Create shadow shapes → Apply with hard edges

Details: Textures → Facial details → Hair highlights → Clothing refinements

Highlights: Selective brightness on skin/hair/metallics (minimal, strategic)

Line Cleanup: Inspect at 200-400% zoom → Refine edges → Verify closures

Quality Check: Run through ALL checklists (Parts 7 & 9) → Verify compliance

Export: PNG transparent background → 3000x4000px minimum → Clean edges

SECTION K: CRITICAL REMINDERS

CHARACTER TYPE RULES:

NORMAL CHARACTERS MUST HAVE: ✅ Warm healthy skin tones (peach/brown/tan)
✅ Eyes with catch light highlights (life/spark)
✅ Natural approachable expressions
✅ Clean well-maintained appearance
✅ Vibrant color palette (70-90% saturation)
✅ Standard shadow coverage (30-40%)

HORROR CHARACTERS MUST HAVE: ✅ Desaturated grey/blue/green skin (30-50% saturation)
✅ Eyes with NO catch light (dead stare)
✅ Emotionless or subtly wrong expression
✅ Facial asymmetry (3-5% difference, subtle)
✅ Heavy under-eye darkness (purple-brown circles)
✅ Skin mottling/uneven tone/veins visible
✅ Unkempt heavy hair
✅ Dark worn clothing with ambiguous stains
✅ Dramatic shadows (40-60% coverage)
✅ NO gore, monsters, or obvious horror - psychological only

ABSOLUTE PROHIBITIONS:

❌ 3D rendering (volumetric, depth simulation, realistic lighting)
❌ Gradients in shading (must be hard-edge cel-shading only)
❌ Angled views (must be perfect 0° front-facing)
❌ Backgrounds (must be fully transparent always)
❌ Gender ambiguity (body structure must clearly indicate male/female)
❌ Sketchy lines (must be clean vector-quality)
❌ Over-saturation (normal) or over-designing (horror)
❌ Horror gore/monsters (psychological wrongness only)

SECTION L: QUALITY ASSURANCE MATRIX

FINAL VERIFICATION CHECKLIST:

TECHNICAL COMPLIANCE:

[ ] Resolution: 3000x4000px @ 300 DPI minimum

[ ] Background: 100% transparent PNG

[ ] View: Perfect front-facing, 0° deviation

[ ] Lines: Vector-clean, 2-3px outlines, consistent weights

[ ] Colors: Accurate to specified palettes

[ ] Shading: Hard-edge cel-shading, no gradients

ANATOMICAL ACCURACY:

[ ] Proportions: Correct head-height ratio for age

[ ] Gender: Clearly identifiable from body structure (shoulder-hip ratio)

[ ] Age: Appropriate features (wrinkles for elderly, larger eyes for kids)

[ ] Posture: Straight, centered, symmetrical

[ ] Limbs: Proper length and positioning

STYLE CONSISTENCY:

[ ] 2D flat aesthetic maintained

[ ] Line quality professional throughout

[ ] Color harmony appropriate to character type

[ ] Detail level balanced (not over or under)

[ ] Matches "Kahaani Monday" / mobile game art style

CHARACTER TYPE SPECIFIC:

If Normal:

[ ] Warm skin tones

[ ] Catch light in eyes

[ ] Natural expression

[ ] Vibrant colors

[ ] Clean appearance

If Horror:

[ ] Desaturated skin

[ ] NO catch light (dead eyes)

[ ] Emotionless/wrong expression

[ ] Asymmetry present

[ ] Dark worn clothing

[ ] Stains present (1-3, ambiguous)

[ ] Dramatic shadows

[ ] NO gore/monsters

PROFESSIONAL STANDARDS:

[ ] Appears hand-crafted by elite artist

[ ] Zero AI-generation indicators

[ ] Adobe Illustrator/Photoshop quality level

[ ] Commercial publication ready

[ ] Unique, memorable design

[ ] Narrative supported by visual choices

SECTION M: EXAMPLE USAGE FRAMEWORK

HOW USER WILL PROVIDE INPUT:

CHARACTER TYPE: [Normal/Horror] AGE: [Kids/Young Adult/Middle Age/Elderly] GENDER: [Male/Female] SKIN TONE: [Light/Medium/Dark + specific ethnicity if relevant] HAIR: [Color, length, style] EYES: [Color] FACIAL FEATURES: [Glasses, facial hair, scars, distinctive features] EXPRESSION: [Neutral, smiling, serious, etc.] CLOTHING: - [Upper body garment description] - [Lower body garment description] - [Footwear description] - [Colors, patterns, condition] ACCESSORIES: [List of items with positions] SPECIAL NOTES: [Any specific requirements, profession, personality traits] 

SYSTEM WILL EXECUTE:

Parse and classify all inputs

Apply appropriate rule sets (Normal vs Horror, Age, Gender)

Select exact color codes from palettes

Construct character following 10-step workflow

Verify against ALL quality checklists

Deliver transparent PNG at specified resolution

SECTION N: ADVANCED CONSIDERATIONS

DIFFERENTIATION STRATEGIES:

To create unique characters within consistent style:

Vary face shapes (oval, round, square, heart, diamond)

Adjust feature sizes (large vs small nose, wide vs narrow eyes)

Modify builds (slim, average, athletic, heavy, tall, short)

Change expression subtly (serious, kind, confident, tired)

Use distinct color palettes per character

Vary clothing styles (formal, casual, traditional, professional)

Select unique accessories that tell story

CULTURAL SENSITIVITY:

Research authentic cultural clothing before rendering

Represent traditional garments respectfully and accurately

Use appropriate colors and patterns for culture

Include culturally specific accessories correctly

Avoid stereotypes or caricatures

Consult references for unfamiliar cultures

EDGE CASES:

Prosthetics/Disabilities: Render devices in same 2D style, integrate naturally

Fantasy Elements: Keep grounded (pointed ears ok, glowing effects NOT ok)

Historical Clothing: Simplify period garments to style level

Tattoos/Scars: Simplified designs, 1-1.5px lines, follow body contours

Androgynous: Balanced proportions, mixed gender indicators

SECTION O: TROUBLESHOOTING QUICK FIXES

Character looks flat: Increase shadow darkness (30-35%), expand coverage (40%), verify light direction consistency

Looks too cartoony: Reduce line weight to 2-3px, add anatomical detail, mature color palette, subtle expressions

Gender unclear: Verify shoulder-hip ratio (CRITICAL), check chest structure (female MUST have bust), facial features differentiation

Horror too obvious: Reduce extreme elements, increase subtlety (3-5% asymmetry only), remove monster features, focus on psychological wrongness

Clothing painted-on: Add 5+ fold lines per area, show fabric weight, layer correctly, proper fit variation

Accessories floating: Attach with contact points, show weight/compression, cast shadows on body, interaction marks

SECTION P: STYLE REFERENCES SUMMARY

"KAHAANI MONDAY" STYLE CHARACTERISTICS:

✅ 2D animated comic/graphic novel aesthetic
✅ Mobile horror game character art quality
✅ Strong black outlines (2-3px), bold linework
✅ Cel-shaded flat color blocks with strategic shadows
✅ Desaturated tones (horror) or vibrant tones (normal)
✅ Expressive faces (normal) or unsettling blank (horror)
✅ Textured painterly shading with hard edges
✅ Front-facing poses, static presentation
✅ Dramatic atmospheric lighting
✅ Professional digital illustration standard
✅ Clean, readable at any scale
✅ Story-driven character design

SECTION Q: FINAL INTEGRATION STATEMENT

COMPLETE SYSTEM OPERATION:

This prompt system integrates 10 comprehensive parts covering every aspect of professional 2D character illustration:

Foundational Framework - Core philosophy, technical specs, proportions

Linework Architecture - Line weights, quality standards, construction order

Color & Shading Systems - Palettes, cel-shading technique, application

Horror Specifications - Psychological horror design, intensity levels, wrongness

Clothing Construction - Garment types, folds, fabrics, professional uniforms

Accessories & Equipment - Footwear, jewelry, bags, tools, cultural items

Workflow & Quality Control - 10-step process, verification checklists

Example Prompts - Real-world applications, variations, practical scenarios

Advanced Techniques - Differentiation, troubleshooting, expert refinements

Master Integration - Complete summary, quick reference, final checklist

The result: A God-tier professional prompt system capable of generating Adobe Illustrator/Photoshop quality 2D characters that are:

Anatomically accurate

Style-consistent

Commercially viable

Culturally respectful

Technically flawless

Narratively compelling

Horror or Normal type specific

Age and gender appropriate

Uniquely memorable

Indistinguishable from professional human artist work

SECTION R: PROMPT ACTIVATION PROTOCOL

WHEN USER PROVIDES CHARACTER DESCRIPTION:

EXECUTE SEQUENCE: 1. READ user input completely 2. CLASSIFY character type (Normal/Horror) 3. IDENTIFY age category, gender, ethnicity 4. SELECT appropriate color palettes from Part 3 5. APPLY proportional system from Part 1 6. CONSTRUCT using workflow from Part 7 7. INTEGRATE clothing from Part 5 8. ADD accessories from Part 6 9. APPLY type-specific rules (Part 4 if horror) 10. VERIFY against checklists Parts 7, 9, 10 11. RENDER at 3000x4000px, transparent background 12. DELIVER professional-quality 2D character illustration STYLE REFERENCE: "Kahaani Monday" mobile horror game aesthetic OUTPUT: PNG with alpha transparency, front-facing, cel-shaded, 2D flat QUALITY STANDARD: Adobe Illustrator/Photoshop professional artist level )",
    height=250,
    placeholder="Paste your first 10,000-word prompt here"
)

prompt_2 = st.text_area(
    "Prompt 2 (​[SYSTEM DIRECTIVE: ULTRA-ELITE ARTIST PROTOCOL]
Act as a world-class Lead Character Designer at a top-tier studio (Adobe Creative Cloud Master). Your mission is to generate a singular, high-fidelity 2D character portrait that defies AI-typical "flatness" and instead exhibits the hand-crafted mastery of Adobe Illustrator’s vector precision and Adobe Photoshop’s advanced digital painting.
​1. PRIMARY STRICTURES (NON-NEGOTIABLE):
​VIEWPOINT: Absolute, strict Front-Facing View. The character must be perfectly symmetrical or near-symmetrical in alignment with the "Straight View" camera. No 3/4 views, no profiles, no dynamic tilts.
​BACKGROUND: Absolute Transparent Background (Alpha Channel). The character must exist in a void, ready for professional compositing. Zero environmental artifacts, zero floor shadows, zero background props.
​DIMENSIONALITY: Pure 2D Illustration. Strictly avoid 3D renders, plastic textures, or CGI depth. The depth must be achieved through professional 2D shading, tracing, and atmospheric layering.
​ARTISTIC TOOLS REPLICATION:
​Illustrator Mode: Utilize variable-width strokes. Lines must be clean, tapered, and deliberate. Every contour must look like a hand-drawn path created with the Pen Tool (P).
​Photoshop Mode: Layered shading with a mix of hard-edged cell shading and soft airbrushed gradients. Use "Multiply" and "Overlay" blending mode logic for realistic color integration.
​2. ANATOMICAL LOGIC (GENDER & AGE SPECIALIZATION):
The prompt must strictly adhere to biological and structural differences between categories.
​MALE FRAME: Professional masculine anatomy. Broader shoulders, trapezoidal torso, sharp and defined jawlines, thicker necks, and prominent brow ridges. Muscle definition is rendered with subtle line-tracing, not caricature.
​FEMALE FRAME: Professional feminine anatomy. Strict adherence to an hourglass or pear-shaped silhouette where applicable. Shoulders are narrower than the hips. Define the bust with anatomical accuracy and professional restraint. The waist-to-hip ratio must be distinct and elegant. Facial features are refined with softer jawlines and more prominent zygomatic (cheek) bones.
​KIDS (JUVENILE): Proportions must follow the 3-to-5 head-height rule. Larger eyes relative to the face, rounded cheeks (buccal fat), softer jawlines, and shorter limbs. Shading is bright and smooth to imply youth.
​OLD AGE (SENIOR): Introduce realistic skin-folding, drooping eyelids (ptosis), and nasolabial folds. Textures must include professional "crackle" and age spots using Photoshop "Grain" and "Noise" brushes. Posture should reflect a slight skeletal compression.
​3. THE "ELITE" SHADING & TRACING ENGINE:
​LINE ART (TRACING): Execute a "Variable Weight Stroke." The outer silhouette (holding line) should be slightly thicker (2pt) while internal details like eyes and lips use micro-fine lines (0.25pt). Every line must be "traced" with the precision of a professional vector artist.
​LIGHTING LOGIC: Direct front-lighting with subtle "Ambient Occlusion" in the crevices (corners of the eyes, beneath the nostrils, under the chin). This creates a "Professional Elite" look that prevents the character from looking like a flat sticker.
​SKIN RENDERING: Avoid uniform color. Skin must show "Micro-Variation." For Normal characters, use healthy subsurface scattering (SSS) hints in the ears and nose. For Horror characters, use the "Sickly Decay" palette analyzed from the reference images.
​4. INTERFACE & PROMPT INPUT STRUCTURE:
User will provide: [Character Description], [Category: Kids/Male/Female/Old], and [Type: Horror/Normal].
​IF HORROR: Trigger the "Psychological Uncanny" module. Focus on micro-asymmetry, light-absorbing pupils, and "frozen mid-breath" expressions.
​IF NORMAL: Trigger the "Cinematic Life" module. Focus on specular highlights in the eyes, vibrant (yet desaturated) professional palettes, and approachable, high-end commercial art aesthetics.
​5. ART STYLE SPECIFICATION:
The final output must resemble a high-budget 2D animation frame or an elite webcomic asset. The style is "Semi-Realistic Anime-Fusion." It maintains the clean lines of high-end Japanese production (like Mappa or Ufotable) but adds the professional Western digital painting textures of Adobe Photoshop.

​[MODULE: ELITE HORROR RENDERING]
When the [Type: Horror] flag is activated, the system must shift from "Commercial Beauty" to "Cinematic Dread." The goal is not "jump-scare" horror, but the deep, professional psychological horror found in elite Adobe-rendered motion comics.
​1. THE "UNCANNY VALLEY" FACIAL LOGIC:
​Micro-Asymmetry (The "Wrong" Feel): Strictly avoid perfect facial symmetry. Implement a 2-3% deviation in eye-level or mouth-corner alignment. One pupil should be slightly more dilated than the other. This creates a subconscious "threat" response in the viewer.
​The "Stillness" Directive: The expression must be "Interrupted Life." No screaming, no snarling. The mouth should be slightly parted as if the character stopped breathing mid-sentence. Lips must show a "Dry-Cracked" texture using Photoshop's Stipple Brush effects.
​Ocular Dread (Aankhon ki Geherayi): * Refractive Void: Unlike normal characters, horror characters have zero catch-lights (white dots) in their eyes. The pupils must be rendered as "Light-Absorbing Wells."
​Sclera Decay: The whites of the eyes (Sclera) should not be white; they must use a palette of Muted Yellow, Pale Grey, and Veined Crimson. Use ultra-fine 0.1pt lines to trace broken capillaries.
​2. SKIN PATHOLOGY & COLOR DISCIPLINE:
​The Sickly Palette: Using the deep analysis of the user's reference images, the skin tone must follow a Desaturated Cyan-Grey Base.
​Shadow Tones: Use Deep Plum, Olive Drab, and Charcoal for the hollows of the cheeks and the orbital bone (eye sockets).
​Subcutaneous Mapping: Trace faint, blue-green veins just beneath the skin surface, especially around the temples and neck. This must look like a professional Photoshop layer with 30% opacity.
​Texture Overlays: Apply a "Post-Mortem Grain." The skin should look cold and matte, avoiding any "AI-plastic" shine. Incorporate subtle "Pore-Level Detail" that looks hand-stippled.
​3. TRAUMA & DECAY DETAILING (STRICT ADOBE ARTIST STYLE):
​Fluid Dynamics (Khoon aur Laar): If the description includes blood or wounds, it must be rendered with High Viscosity. Blood is not bright red; it is Oxidized Crimson/Black-Red. Use Photoshop's Liquid Tool logic to show gravity-defying drips that cling to the skin texture.
​Anatomical Rupture: For decapitated or wounded characters (per reference images), the cross-section of muscle and bone must be "Cleanly Illustrated." Use a mix of hard-edged vector lines for bone and soft, wet-look shading for exposed tissue.
​Stitched & Bound: Any stitches or sutures must have "Skin Tension." The skin around the thread must be slightly puckered and red-tinted (Adobe Photoshop Burn Tool simulation).
​4. HAIR & FILAMENT LOGIC:
​The "Heavy" Hair Look: Hair should not look bouncy or healthy. It must look Damp, Weighted, and Clumped.
​Tracing Rules: Use Illustrator's Tapered Stroke to create individual "Straggler" hairs that cross the face irregularly. This breaks the "clean" look and adds to the disheveled horror aesthetic.
​Coloring: Use flat, dark bases with extremely thin, high-contrast highlights to simulate greasy or sweat-soaked hair.
​5. CLOTHING DISTRESS (THE "DIRTY" LAYER):
​Fabric Decay: Clothes must look "Worn for Centuries." Use a "Grunge Overlay" texture.
​Shadow Weight: Clothing folds must have deep, heavy shadows that hide the character's true form, creating mystery.
​Stain Logic: Incorporate "Ambient Grime" around collars and cuffs using a low-opacity Sponge Brush effect.
​6. SUMMARY OF HORROR "TRACING":
Every horror feature—from a sunken cheek to a missing limb—must be "Traced" with a sharp black or deep-brown contour. This ensures the character looks like a Professional 2D Asset and not a blurry AI mess. The contrast between the sharp lines and the sickly, muddy shading is the key to this "Ultra-Elite" style.

[MODULE: ANATOMICAL PRECISION & GENDER DYNAMICS]
When the [Type: Normal] flag is activated, the system must switch to a "High-End Animation Studio" output. The focus is on clean, healthy proportions, professional skin luster, and strict biological accuracy as per the user's Strict Rules.
​1. THE FEMALE BODY ARCHITECTURE (STRICT RULE):
​Silhouette & Frame: Implement a professional feminine skeletal structure. Shoulders must be softer and narrower than the hips to create a classic 2D "Hourglass" or "Pear" profile.
​Torso & Bust Rendering: Render the bust with anatomical realism and professional restraint, ensuring it aligns with the ribcage naturally. The waistline must be clearly defined with a sharp "Adobe Illustrator" curve leading into the hips.
​Facial Features: High-fashion "Point-Vector" precision. Use tapered lines for eyelashes and a soft "Photoshop Blush" layer on the cheekbones. The jawline should be smooth and elegant, avoiding the heavy angularity of the male frame.
​2. THE MALE BODY ARCHITECTURE (STRICT RULE):
​The "Inverted Triangle" Frame: Broad, sturdy shoulders that taper down to a narrower waist. The neck must be thicker and more muscular, showing the "Sternocleidomastoid" muscle traced with a fine 0.2pt line.
​Facial Geometry: Focus on "Hard Surfaces." A sharp, defined jawline (Mandible) and a prominent brow ridge. Use "Adobe Photoshop's" Chisel Brush logic to create shadows under the jaw, giving it a heavy, masculine weight.
​Traced Details: Add subtle hand-traced lines for the collarbones and forearm tendons to imply strength without looking like a 3D model.
​3. THE JUVENILE (KIDS) MODULE:
​The 1:4 Head Ratio: Proportions must reflect childhood. The head should be slightly larger in proportion to the torso.
​Feature Roundness: Every "Adobe Illustrator" path must be rounded. No sharp angles on the face. Large, expressive irises with "Triple-Layer Specular Highlights" to convey innocence and life.
​Shading Style: Use a "Peachy-Pink" palette for joints (elbows, knees) and cheeks to simulate high-budget 2D anime styles. Skin should look flawless and soft.
​4. THE SENIOR (OLD AGE) MODULE:
​Skeletal Gravity: Show the effect of age on posture. The shoulders should be slightly more rounded, and the neck forward-leaning.
​Skin Topography: This is where "Adobe Photoshop" textures are critical. Use a custom Noise/Grain Overlay to create age spots and fine-line wrinkles.
​Strict Tracing: Trace the "Crows-feet" around the eyes and "Nasolabial Folds" around the mouth with extremely thin, variable-width lines. This ensures the character looks "Old" through professional illustration, not through a "filter."
​5. ELITE SKIN & LIGHTING (NORMAL TYPE):
​The "Healthy Glow": Unlike the horror module, skin here must have Subsurface Scattering. Use warm orange/red tones in the "Ambient Occlusion" areas where skin meets skin (e.g., behind the ears).
​Digital Trace-Shading: Use "Cell Shading" for the primary shadows and a "Gaussian Blur" gradient for the secondary transitions. This creates a "Professional Elite" look that mimics a hand-painted Adobe Photoshop masterpiece.
​Ocular Life: Eyes must contain a "Catchlight" (Primary highlight) and a "Refracted Glow" in the lower iris. This makes the character appear conscious and approachable.
​6. CLOTHING PHYSICS:
​Illustrator Folds: Folds in the fabric must follow the "Point of Tension" (e.g., where the arm bends). Use sharp, clean lines for the folds and soft gradients for the shadows within those folds.
​Material Distinction: Use different "Photoshop Brush" textures for silk (high shine), wool (matte/grainy), and cotton (flat/soft).

​[MODULE: HAIR ARCHITECTURE & FILAMENT RENDERING]
In professional 2D character design, hair is not a single mass; it is a collection of "Shape Groups" and "Individual Strands." This module dictates how the system must "Trace" and "Shade" hair to ensure it looks hand-drawn by an Adobe master, avoiding the blurry or "painted-on" look of standard AI.
​1. THE ILLUSTRATOR PATH LOGIC (TRACING):
​Variable Stroke Width: Hair outlines must use Pressure-Sensitive Path Logic. The base of a hair clump should be thicker (0.75pt), tapering to a razor-sharp point (0.1pt) at the tip.
​Layered Clumping: Divide the hair into three layers: Primary Masses (the main shape), Secondary Tresses (smaller groups that add volume), and Tertiary Flyaways (stray strands).
​Silhouette Integrity: The outer edge of the hair must be a clean, continuous vector line. Internal detail lines should be sparser to allow the shading to define the form.
​2. THE PHOTOSHOP DEPTH ENGINE (SHADING):
​Three-Tone Coloring: Every hair color must consist of a Base Mid-tone, a Deep Shadow (Multiply layer), and a Specular Highlight (Screen/Overlay layer).
​Ambient Occlusion in Roots: Apply deep, localized shadows where the hair meets the scalp and behind the ears. This "Photoshop Burn" effect gives the hair weight and anchors it to the head.
​Anisotropic Highlights: Highlights should follow the "Halo" or "Ring" pattern typical of professional anime and digital painting, appearing across the curve of the head to simulate light reflecting off smooth surfaces.
​3. TEXTURE & MATERIALITY (NORMAL VS. HORROR):
​Normal Character Hair: Focus on Silky Luster. Use soft gradients within the tresses. The "Flyaway" strands should look intentional and elegant. For female characters, emphasize flow and bounce; for male characters, emphasize structure and direction.
​Horror Character Hair (Strict Rule): Focus on Grime and Weight.
​The "Wet Look": Use high-contrast, sharp highlights to simulate sweat or grease.
​Clumping: Instead of silky flow, hair should be "matted" into jagged, irregular chunks.
​Straggler Logic: Draw thin, erratic lines that partially obscure the eyes or mouth to increase the "Uncanny" feel.
​4. AGE-SPECIFIC HAIR RENDERING:
​Kids: Soft, fine, and wispy. Use a "High-Key" palette with minimal shadows. Lines should be very light and rounded.
​Old Age (Senior): Focus on Thinner Density. Trace the scalp through the hair in certain areas. Use a "Matte Texture" with zero luster. Lines should be more erratic and "dry" looking (using a Photoshop Charcoal or Dry Media brush simulation).
​Male Facial Hair: Beards and stubble must be "Traced" as individual micro-lines, not a flat color block. Use "Stipple Shading" at the edges of the beard to show skin-to-hair transition.
​5. THE "ADOBE ARTIST" FINISHING TOUCH:
​Color Jitter: Introduce microscopic color variations (e.g., a few strands of slightly warmer or cooler brown in a brunette head). This prevents the "flat AI fill" look.
​Edge Refinement: Ensure no "Haloing" or "Ghosting" occurs at the hair edges. Every strand must terminate cleanly against the transparent background.
​6. INTERACTION WITH LIGHT:
​Rim Lighting (Subtle): Apply a very thin (0.5pt) light-colored line on the side of the hair opposite the main light source to separate it from the void.
​Translucency: For thin strands (especially in kids or blondes), the edges should have a slight "Light Bleed" effect where they meet the background

[MODULE: MICRO-DETAIL FACIAL SENSORS]
In elite Adobe Illustrator and Photoshop workflows, the eyes and mouth define the "Soul" or the "Horror." This module dictates the exact tracing and shading required to make these features look hand-painted and biologically accurate.
​1. OCULAR ARCHITECTURE (THE EYES):
​The Sclera (White of the Eye): Never use pure white. For Normal characters, use a "warm ivory" or "pale blue-grey" with a soft Photoshop gradient to show the eyeball's curvature. For Horror characters, use "sickly yellow" or "bruised purple" undertones with micro-veins traced at 0.05pt thickness.
​The Iris & Pupil (Elite Shading):
​Normal Logic: Use a "Radial Gradient." The top of the iris (under the eyelid) is darker due to the shadow of the lashes. The bottom of the iris has a "Refractive Glow."
​Horror Logic (Strict Rule): Remove the refractive glow. The iris should appear "Flat" or "Sunken." Pupils must be dilated to an unnatural size or constricted to pinpricks, absorbing all light with zero specular reflection.
​Lid & Lash Tracing: Upper eyelids must have a thicker "Illustrator Path" to simulate the shadow of eyelashes. Lower lashes should be rendered as tiny, individual "tapered strokes," not a solid line.
​The Tear Duct (Canthus): Add a microscopic "High-Gloss" dot of moisture (Specular Highlight) in the corner of the eye to differentiate the biological tissue from the skin.
​2. ORAL ARCHITECTURE (THE MOUTH & LIPS):
​Lip Mapping (Strict Gender & Age Rule):
​Female: Sharp, clean "Cupid's Bow" definition. Use a "Satin-Finish" gradient with a vertical "lip-line" texture (fine 0.1pt strokes) to show realistic skin stretching.
​Male: More rectangular and matte. The transition between the lip and the skin should be less defined (softer blending) to avoid a "lipstick" look unless specified.
​Old Age: Incorporate "Vertical Fissures" (cracks) and a loss of volume. The corners of the mouth (Commissures) should have a slight downward "Gravity Droop."
​The "Slightly Parted" Rule: To match your professional reference style, the lips should be frozen mid-breath. This requires rendering the "Internal Oral Void."
​Teeth & Gum Rendering: Teeth should not be individual white blocks. They should be rendered as a "Unified Mass" with subtle vertical "Ambient Occlusion" lines between them. For horror, apply "Gingival Recession" (showing more of the tooth root) with a yellowish-brown decay gradient.
​3. MICRO-EXPRESSION LOGIC (THE "WRONG" LOOK):
​Asymmetrical Tension: In the horror module, one corner of the mouth must be 1-2mm higher than the other. The "Nasolabial Fold" (the line from nose to mouth) should be deeper on only one side.
​Philtrum Definition: The groove above the upper lip must be traced with a soft "Photoshop shadow" to give the face a 3D structural feel while remaining a 2D illustration.
​4. MOISTURE & GLOSS CONTROL:
​Saliva Logic (Horror): Use a "Dodge Tool" effect to create thin, glistening strings of moisture if the mouth is open. This must look like a high-gloss Photoshop layer with 80% opacity.
​Healthy Luster (Normal): A single, horizontal "Swipe Highlight" on the lower lip to indicate health and hydration.
​5. THE "ADOBE TRACING" FINISH:
​Every feature must be enclosed in a "Pressure-Sensitive Stroke." The line should vanish (0pt thickness) at the highlights and thicken (1pt thickness) at the deepest shadow points. This creates the "Elite Illustrator" look you requested

​[MODULE: DERMAL RENDERING & CHROMATIC DISCIPLINE]
This module governs the "Surface Quality" of the character. It prevents the skin from looking like plastic and ensures the texture matches the elite standard of an Adobe artist who uses customized grain and texture brushes.
​1. THE "ADOBE PHOTOSHOP" SKIN ENGINE:
​Base Layering: Start with a mid-tone base. Never use a single flat color. Use the "Sponge Tool" logic to vary the saturation across the face.
​Subsurface Scattering (SSS): In Normal characters, apply a "Warm Red-Orange" tint to areas where the skin is thin (ears, nostrils, eyelids). This simulates light passing through tissue.
​Micro-Texture (Grain): Apply a global 3% Film Grain overlay. This breaks the digital smoothness and gives it the "Printed Comic" or "Motion Picture" texture seen in professional 2D assets.
​2. COLOR PALETTE DISCIPLINE:
​Normal Characters: Use a "Cinematic Neutral" palette. Avoid neon or hyper-saturated skin. Shadows should be "Warm" (using Burnt Sienna or Deep Peach) to maintain a healthy appearance.
​Horror Characters (Strict Rule): Use a "Necrotic & Desaturated" palette.
​The "Bruise" Logic: Instead of black shadows, use Muted Indigo, Olive Green, and Dirty Violet.
​Venous Mapping: Trace a network of faint, "Deep Teal" veins at 15% opacity under the skin surface of the forehead and neck.
​Color Zoning: The area around the eyes must be slightly "Red-Brown" (exhaustion), and the area around the mouth slightly "Grey-Blue" (lack of oxygen/blood flow).
​3. SKIN TOPOGRAPHY & IMPERFECTIONS:
​The "Anti-Filter" Rule: The skin must have "Human Realism." This includes microscopic moles, slight unevenness in tone, and realistic "Pore Mapping" in the T-zone (nose and forehead).
​Specular Highlights: Use a "Hard-Edge" white highlight only on the tip of the nose and the "Philtrum" (above the lip) to indicate natural skin oils. All other highlights must be "Soft-Glow" (Photoshop Gaussian Blur style).
​Tracing Wrinkles: * Young/Kids: Zero wrinkles, only soft "folds" at the joints.
​Adults: Define the "Nasolabial Folds" (smile lines) with a very thin, light-colored highlight line alongside a thin dark shadow line.
​Old Age: Deep "Dermal Fissures." Use a "Bevel and Emboss" logic where each wrinkle has a shadow side and a light-catching edge.
​4. GENDER-SPECIFIC SKIN TEXTURE:
​Female: Focus on "Luminous Matte." The shading should be seamless with very few hard lines. The transition from the cheekbone to the jaw should be a smooth gradient.
​Male: Focus on "Rugged Definition." Use more "Hard-Edge" shadows on the jawline and brow. If stubble is present, it must be rendered as a "Stipple Overlay," not a solid grey mass.
​5. TRACING THE SILHOUETTE (THE "ILLUSTRATOR" OUTLINE):
​The skin must be contained within a Clean Vector Contour. The color of the outline should not be pure black; it should be a "Darker Version" of the skin tone (e.g., Deep Mahogany for tan skin, Charcoal for pale skin). This is a hallmark of elite Adobe professional art.
​6. AMBIENT OCCLUSION (THE "ELITE" SECRET):
​Apply the deepest shadows in the "Micro-Gaps": where the hair touches the forehead, under the lower lip, inside the nostrils, and the deep corner of the eyes. This creates "Weight" and "Volume" without needing 3D tools.

​[MODULE: LIGHT PHYSICS & ATMOSPHERIC VOLUME]
This module defines how light interacts with the character's surface. To avoid the "AI-Flat" look, we must simulate the behavior of light as if it were being digitally painted in layers using Adobe Photoshop’s Blending Modes (Multiply, Screen, Overlay).
​1. PRIMARY LIGHT SOURCE (FRONT-FACING DIRECT):
​The "Key Light": Light must originate from a "Central Frontal" position to satisfy the Strict Front View rule. This light should create a subtle "T-shape" of brightness across the forehead and down the bridge of the nose.
​Falloff Logic: Use a Soft-Edge Gradient falloff. The light should be brightest at the center of the face and gradually darken toward the ears and hairline, creating the illusion of a curved 3D surface on a 2D plane.
​2. THE "SHADOW TRACING" RULE (AMBIENT OCCLUSION):
​Micro-Shadows: This is the "Professional Secret." Use ultra-dark, high-saturation "Contact Shadows" in the narrowest gaps:
​Behind the earlobes.
​Under the tip of the nose (Columella).
​Between the lips (The "Oral Void").
​Under the upper eyelid (creating the "Eyelash Shadow").
​Color of Shadows: NEVER use pure black for shadows in Normal characters. Use a Deep Burgundy, Navy, or Forest Green depending on the character’s skin undertone. This is the "Adobe Artist" method for "Rich Shadows."
​3. RIM LIGHTING & EDGE DEFINITION (THE ALPHA SEPARATOR):
​Back-Rim Logic: To ensure the character "pops" against the Strictly Transparent Background, apply a microscopic (0.5pt to 1pt) "Rim Light" or "Kicker" along the silhouette of the hair and shoulders.
​Color Temperature: In Normal characters, use a cool white or pale gold. In Horror characters, the rim light should be a "Ghostly Blue" or "Sickly Green" to enhance the psychological dread.
​4. CAST SHADOWS VS. FORM SHADOWS:
​Anatomical Cast Shadows: The head must cast a clear, clean-edged shadow onto the neck. This shadow should be "Traced" with an Illustrator-style path to maintain the 2D illustration look.
​Form Shadows: Use soft Photoshop brushes to create the "Form" of the muscles (on males) or the "Curvature" of the bust/waist (on females). This is what defines the Strict Body Shape rules you requested.
​5. SPECULAR HIGHLIGHTS (THE "GLOSS" LAYER):
​Surface Materiality: Use different "Gloss Levels" for different parts of the character:
​Eyes: High-Gloss (Sharp, white dots).
​Lips: Satin-Gloss (Soft, elongated streaks).
​Skin: Matte-Gloss (Broad, low-opacity glows).
​Metal/Armor (if any): Mirror-Gloss (High contrast).
​6. THE "ADOBE TRACING" FINISH (SHADOW EDGES):
​Ensure that where a shadow meets a light area, there is a "Transition Zone." Professional artists use a "Slightly Saturated Edge" (e.g., a thin orange line between a brown shadow and peach skin) to simulate Subsurface Scattering. The prompt must hard-code this "Saturated Fringe" logic.

​[MODULE: TEXTILE ARCHITECTURE & DRAPE DYNAMICS]
In elite character design, clothing is not just a "color fill"—it is a structural layer that defines the character's silhouette and status. This module ensures fabric behaves according to professional physics while maintaining the Strict Front View and Gender-Specific Body Shapes.
​1. ANATOMICAL DRAPE (STRICT BODY RULE):
​Female Clothing Logic: Fabric must follow the "Contour-Mapping" of the female frame. In professional Adobe-style art, this means using Tension Lines that originate from the bust and hips. Clothing should not "hang flat"; it must show the underlying volume of the waist-to-hip ratio using Illustrator's clean, sweeping curves.
​Male Clothing Logic: Focus on "Structural Weight." Fabric should hang from the shoulders (the primary tension point). Use thicker "Adobe Illustrator" paths for collars, lapels, and shoulder seams to emphasize the broad masculine frame.
​Kids & Seniors: For kids, clothing should look slightly oversized or soft-edged. For seniors, fabric should show "Gravity Sag," with more folds accumulating at the waist and wrists.
​2. THE "ILLUSTRATOR" FOLD ENGINE (TRACING):
​Types of Folds: The prompt must differentiate between Pipe Folds (cylindrical), Zig-zag Folds (at joints), and Drape Folds (hanging).
​Line Weight Variation: The "Deepest Part" of a fold must have a thicker vector line (0.8pt), while the "Apex" (the part catching the light) should have no line or a very thin, light-colored line (0.2pt). This is the hallmark of elite 2D tracing.
​3. THE "PHOTOSHOP" TEXTURE ENGINE (SHADING):
​Material-Specific Rendering:
​Cotton/Matte: Use flat cell-shading with a 5% "Photoshop Noise" overlay to simulate fiber.
​Silk/Satin: Use high-contrast gradients with sharp, white "Specular Streaks" along the ridges of the folds.
​Leather/Vinyl: Use "Hard-Edge" highlights and deep, nearly-black shadows to create a reflective, tough surface.
​Wool/Heavy Fabric: Use a "Textured Brush" effect on the shadow edges to show the "fuzz" or thickness of the material.
​4. HORROR VS. NORMAL FABRIC (STRICT RULE):
​Normal Type: Clothes must look "Freshly Pressed" or "Cleanly Worn." Shadows are crisp, and the colors are balanced. Use a "Soft Glow" layer on the highlights to give a premium, cinematic feel.
​Horror Type: Activate the "Distress Protocol."
​The "Weight" of Filth: Clothing must appear heavy, as if soaked in moisture or oil. Use "Multiply" layers in Photoshop to create damp patches around the collar and underarms.
​Fraying & Micro-Tears: Use 0.1pt "Scattered Lines" at the edges of the sleeves to show wear.
​Stain Logic: Add "Ambient Grime" using a low-opacity, textured "Sponge Brush" effect. These stains must look "Set-in" and permanent, not like a surface splatter.
​5. HARD SURFACE ELEMENTS (BUTTONS, ZIPPERS, ACCESSORIES):
​Any metal or plastic accessories must be rendered with "Precision Vector Logic." Buttons should have a "Bevel and Emboss" look with a tiny "Photoshop Dodge" highlight on the top edge to make them look solid.
​6. THE "TRANSPARENT EDGE" PROTOCOL:
​Ensure that the outer silhouette of the clothing is razor-sharp. There must be zero "bleeding" or "blurring" into the transparent background. The "Alpha Edge" must be a clean vector path, ensuring the character can be placed into any professional composition immediately.

[MODULE: CHROMATIC POLISH & STUDIO MASTERING]
This module ensures that the individual elements (eyes, skin, hair, clothes) are unified into a single, cohesive masterpiece. It applies the final "Post-Production" effects that separate a standard AI image from an Elite Adobe Professional artwork.
​1. COLOR HARMONY & UNIFICATION:
​The "Global Color Grade": Apply a subtle "Color Lookup" (LUT) logic. For Normal characters, use a "warm-cinematic" or "neutral-clean" balance to make the skin tones look healthy. For Horror characters, shift the mid-tones toward "Cold Cyan" or "Sickly Desaturated Green."
​Shadow Tinting: Shadows must never be neutral grey. Use the "Adobe Color Balance" rule: if the light is warm, the shadows must have a cool (Blue/Purple) tint. This adds professional depth and "vibrancy" to the 2D plane.
​2. THE "ELITE" TEXTURE OVERLAY:
​The 2D Grain Protocol: To kill the "plastic" AI look, apply a 2% Fine Grain Noise Layer at the very top of the stack. This mimics the texture of a high-quality animation cell or a printed graphic novel.
​Micro-Canvas Texture: Introduce an almost invisible "Paper Tooth" or "Canvas Fiber" texture using a Photoshop Overlay Blend Mode at 5% opacity. This makes the character look "Hand-Painted" rather than "Computer Generated."
​3. OPTICAL EFFECTS (ELITE POLISH):
​Subtle Chromatic Aberration: Apply a microscopic (0.2px) color fringe (Red/Cyan) only at the outermost edges of the silhouette. This mimics a real camera lens and adds a "High-Budget" feel.
​Soft Bloom/Glow: Apply a very low-radius Gaussian Blur to the brightest highlights (like the sparkle in the eyes or the shine on the lip). This creates a "Dreamy" or "Cinematic" light-bleed effect.
​4. THE "SHARPENED TRACING" FINISH:
​High-Pass Filter Logic: After all shading is done, the "Adobe Illustrator" line-art must be sharpened. Every vector path should be crisp and clear. Use a Unsharp Mask effect on the line-art layer to ensure the "Tracing" is the most prominent feature.
​Alpha Edge Anti-Aliasing: The transition between the character and the Transparent Background must be perfectly smooth. There should be no "Jagged Pixels" (aliasing). The edge must be a "Clean Vector Cut."
​5. DEPTH OF FIELD (2D STYLE):
​Edge Softening: While the face remains sharp, the furthest parts of the character (like the back of the hair or the far shoulder) should be 5% softer. This creates a "Micro-Depth" that guides the viewer's eye to the face (The Ocular Center).
​6. FINAL EXPORT SPECIFICATIONS:
​The final image must appear as a 32-bit Deep Color asset.
​Contrast must be "Studio-Leveled"—meaning the darkest point is a true "Professional Black" (not washed out) and the brightest point is a "Crisp White."

[FINAL SYSTEM OVERRIDE: THE GOD-TIER UNIFICATION]
To achieve the "Adobe Professional" result, all previous modules (Anatomy, Ocular, Oral, Shading, Tracing, and Color Grading) must now be activated through a singular Master Input Structure.
​1. HOW TO MERGE THE 10,000 WORDS:
Combine the text from Part 1 through Part 9 into one continuous "System Instruction" or "Mega-Prompt." When you use it, you must start with the Character Identity Block followed by the Style Enforcement Protocol:
​[Character Identity Block]
​Subject: [Insert Name/Role, e.g., "A weathered soldier" or "A young Victorian girl"]
​Category: [Choose: Male / Female / Kids / Senior]
​Type: [Choose: Normal / Horror]
​Pose: [STRICT RULE: Front-Facing, Straight-View, Symmetric]
​Output: 2D Illustration, Transparent Background, Adobe Master Class Quality.
​2. THE "ELITE" TRIGGER COMMANDS (HINDI-ENGLISH MIX):
In the prompt's final execution, ensure these Micro-Level keywords are present to force the AI to use the analysis from your reference images:
​"Variable Weight Vector Tracing": This ensures the line art looks hand-drawn in Illustrator.
​"Subsurface Scattering & Ambient Occlusion": This forces the "Adobe Photoshop" professional shading.
​"Anatomical Accuracy Protocol": This enforces your strict rules for male/female body shapes.
​"Ocular Micro-Detail": This triggers the deep-eye analysis (reflection for normal, void for horror).
​3. FINAL VALIDATION STEPS:
Before finalizing the image, the prompt instructs the engine to check for:
​Anti-Aliasing: No blurry edges on the transparent background.
​Specular Fidelity: Are the lip and eye highlights in the right place?
​Biological Consistency: Is the waist-to-hip ratio (Female) or shoulder-breadth (Male) consistent with the 10,000-word rules?
​4. THE RESULTING ARTISTIC PHILOSOPHY:
By using this 10-part system, you aren't just asking for a character; you are commanding a virtual Adobe Creative Suite Master to:
​Trace with the precision of an Illustrator Pen Tool.
​Paint with the depth of a 50-layer Photoshop document.
​Grade with the cinematic eye of a Hollywood colorist.
​SUMMARY OF THE 10,000-WORD PROMPT CAPABILITIES:
​Deep Micro-Analysis: Your eyes, lips, and shading rules are now hard-coded into the AI's logic.
​Zero Error Rate: By specifying the view (Front) and background (Transparent), we remove 99% of AI failures.
​Professional Tier: The output will look like a "Normal Character Example" or a "Deep Horror Asset" suitable for high-end gaming or animation.)",
    height=250,
    placeholder="Paste your second 10,000-word prompt here"
)

# =====================================
# USER INPUT
# =====================================

st.subheader("🎭 Character Input")

description = st.text_input(
    "Character Description",
    placeholder="Example: pale woman, long black hair, empty eyes"
)

style = st.radio(
    "Character Style",
    ["Normal", "Horror"]
)

# =====================================
# MODEL LOADING
# =====================================

@st.cache_resource
def load_model():
    model_id = "runwayml/stable-diffusion-v1-5"
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    pipe = pipe.to("cuda" if torch.cuda.is_available() else "cpu")
    return pipe

pipe = load_model()

# =====================================
# PROMPT MIXING LOGIC (BASIC)
# =====================================

def build_final_prompt(p1, p2, desc, style):
    """
    ⚠️ This is a BASIC mixer.
    10k-word prompts will be truncated by the model.
    This is for testing / structure only.
    """

    if style == "Normal":
        style_hint = "Focus on normal human proportions, clean illustration style."
    else:
        style_hint = "Focus on horror elements, unsettling mood, psychological fear."

    final_prompt = f"""
{style_hint}

MASTER PROMPT A:
{p1}

MASTER PROMPT B:
{p2}

USER DESCRIPTION:
{desc}
"""
    return final_prompt


# =====================================
# GENERATION
# =====================================

if st.button("Generate Character"):
    if not prompt_1.strip() or not prompt_2.strip():
        st.error("Please paste BOTH Prompt 1 and Prompt 2.")
    elif not description.strip():
        st.error("Please enter a character description.")
    else:
        final_prompt = build_final_prompt(
            prompt_1,
            prompt_2,
            description,
            style
        )

        with st.spinner("Generating character (this may be slow)..."):
            image = pipe(
                final_prompt,
                num_inference_steps=25,
                guidance_scale=7.5
            ).images[0]

        st.image(image, caption="Generated Character")
        st.success("Character generated!")
