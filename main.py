from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import random
import uvicorn

app = FastAPI(title="AroMi AI Agent API", version="1.0.0")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response Models
class ContentRequest(BaseModel):
    username: str
    topic: str
    language: str

class ContentResponse(BaseModel):
    content: str

class FitnessRequest(BaseModel):
    height: int
    weight: int
    goal: str

class FitnessResponse(BaseModel):
    bmi: float
    category: str
    plan: List[str]

class DiseaseRequest(BaseModel):
    disease: str
    preference: Optional[str] = "No Preference"

class DiseaseResponse(BaseModel):
    recommended: List[str]
    avoid: List[str]

class HealthResponse(BaseModel):
    status: str
    message: str

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    return HealthResponse(
        status="healthy",
        message="AroMi AI backend is running successfully!"
    )

# Content Generation Endpoint
@app.post("/generate", response_model=ContentResponse)
async def generate_content(request: ContentRequest):
    """
    Generate AI-powered content based on topic and language
    """
    try:
        # Simulated AI content generation
        # In production, integrate with actual AI models like GPT
        
        content_templates = {
            "english": {
                "Artificial Intelligence": """# The Future of Artificial Intelligence

Artificial Intelligence (AI) is revolutionizing the way we live and work. From self-driving cars to virtual assistants, AI is becoming an integral part of our daily lives.

## Key Areas of AI Impact:
1. **Healthcare**: AI-powered diagnosis and drug discovery
2. **Education**: Personalized learning experiences
3. **Business**: Automated decision-making and analytics
4. **Environment**: Climate change prediction and solutions

## Challenges Ahead:
- Ethical considerations and bias in AI
- Privacy concerns
- Job displacement vs. job creation

The future of AI is both exciting and challenging. As we continue to develop these technologies, it's crucial to ensure they benefit humanity as a whole.""",
                
                "Climate Change": """# Understanding Climate Change: A Call to Action

Climate change is one of the most pressing issues of our time. The Earth's temperature has risen significantly over the past century, leading to severe environmental consequences.

## Key Facts:
- Global temperatures have risen by 1.1°C since pre-industrial times
- Sea levels are rising at an accelerating rate
- Extreme weather events are becoming more frequent

## What Can We Do?
1. Reduce carbon emissions
2. Switch to renewable energy
3. Practice sustainable living
4. Support environmental policies

Every action counts in the fight against climate change. Together, we can make a difference.""",
                
                "default": """# {topic}

Here's some interesting information about {topic}:

## Key Points:
• {topic} is an important topic in today's world
• It affects various aspects of our lives
• Understanding it better can help us make informed decisions

## Why It Matters:
The significance of {topic} cannot be overstated. It plays a crucial role in shaping our future and the world around us.

## Take Action:
Learn more about {topic} and share this knowledge with others. Every conversation makes a difference!"""
            },
            "hindi": {
                "default": """# {topic} के बारे में जानकारी

{topic} आज के समय का एक महत्वपूर्ण विषय है। आइए जानते हैं इसके बारे में:

## मुख्य बिंदु:
• {topic} हमारे दैनिक जीवन को प्रभावित करता है
• इसकी समझ से हम बेहतर निर्णय ले सकते हैं
• यह भविष्य के लिए महत्वपूर्ण है

## क्यों जरूरी है:
{topic} की अहमियत को कम करके नहीं आंका जा सकता। यह हमारे भविष्य को आकार देने में महत्वपूर्ण भूमिका निभाता है।"""
            }
        }
        
        # Select language template
        lang_key = request.language.lower()
        if lang_key not in content_templates:
            lang_key = "english"
        
        templates = content_templates[lang_key]
        
        # Get content or use default
        if request.topic in templates:
            content = templates[request.topic]
        else:
            default_template = templates.get("default", content_templates["english"]["default"])
            content = default_template.replace("{topic}", request.topic)
        
        return ContentResponse(content=content)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Fitness Planning Endpoint
@app.post("/fitness-plan", response_model=FitnessResponse)
async def get_fitness_plan(request: FitnessRequest):
    """
    Generate personalized fitness plan based on user metrics
    """
    try:
        # Calculate BMI
        height_m = request.height / 100
        bmi = request.weight / (height_m * height_m)
        bmi = round(bmi, 1)
        
        # Determine BMI category
        if bmi < 18.5:
            category = "Underweight"
        elif 18.5 <= bmi < 25:
            category = "Normal weight"
        elif 25 <= bmi < 30:
            category = "Overweight"
        else:
            category = "Obese"
        
        # Generate personalized plan based on goal
        goal = request.goal.lower()
        plan = []
        
        if "lose" in goal:
            plan = [
                "🏃 Cardio: 30-45 minutes daily (running, cycling, swimming)",
                "🥗 Calorie deficit diet (reduce 500 calories from maintenance)",
                "💪 Strength training: 3 times a week",
                "🥤 Drink 3-4 liters of water daily",
                "😴 Get 7-8 hours of sleep",
                "📊 Track your progress weekly"
            ]
        elif "gain" in goal or "muscle" in goal:
            plan = [
                "🏋️ Strength training: 4-5 times a week",
                "🍗 High protein diet (1.6-2.2g protein per kg bodyweight)",
                "📈 Calorie surplus (300-500 calories above maintenance)",
                "🥩 Include lean meats, eggs, dairy, and legumes",
                "💤 Rest and recovery are crucial",
                "📝 Progressive overload in workouts"
            ]
        elif "maintain" in goal:
            plan = [
                "🚶 Active lifestyle: 10,000 steps daily",
                "⚖️ Balanced diet with proper macros",
                "🏋️ Exercise: 3-4 times a week (mix of cardio and strength)",
                "🧘 Include flexibility training",
                "💧 Stay hydrated",
                "📊 Monitor weight weekly"
            ]
        elif "endurance" in goal:
            plan = [
                "🏃 Long-distance cardio: 3-4 times a week",
                "⏱️ Interval training: 2 times a week",
                "💪 Light strength training for muscle endurance",
                "🍝 Complex carbs for sustained energy",
                "🧘 Yoga for flexibility and recovery",
                "📈 Gradually increase intensity"
            ]
        else:  # Get Fit (general)
            plan = [
                "🚶 Start with 20-30 minutes walking daily",
                "💪 Basic bodyweight exercises (push-ups, squats, lunges)",
                "🥗 Eat whole foods, avoid processed items",
                "🧘 Stretch for 10 minutes daily",
                "💧 Drink plenty of water",
                "📈 Gradually increase workout intensity"
            ]
        
        return FitnessResponse(
            bmi=bmi,
            category=category,
            plan=plan
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Disease Diet Planning Endpoint
@app.post("/disease-diet", response_model=DiseaseResponse)
async def get_disease_diet(request: DiseaseRequest):
    """
    Generate dietary recommendations based on health condition
    """
    try:
        disease = request.disease.lower()
        
        # Diet databases
        diet_plans = {
            "diabetes": {
                "recommended": [
                    "Leafy greens (spinach, kale, lettuce)",
                    "Whole grains (oats, quinoa, brown rice)",
                    "Lean proteins (chicken, fish, tofu)",
                    "Berries and citrus fruits",
                    "Nuts and seeds (almonds, walnuts, chia seeds)",
                    "Legumes (beans, lentils, chickpeas)",
                    "Greek yogurt (unsweetened)"
                ],
                "avoid": [
                    "Sugary beverages and sodas",
                    "White bread and refined flour",
                    "Processed snacks and sweets",
                    "Fried foods",
                    "High-sugar fruits (bananas, grapes, mangoes)",
                    "Sweetened breakfast cereals",
                    "Honey, maple syrup, and added sugars"
                ]
            },
            "hypertension": {
                "recommended": [
                    "Bananas and avocados (potassium-rich)",
                    "Leafy greens (spinach, swiss chard)",
                    "Beets and beetroot juice",
                    "Oats and whole grains",
                    "Fatty fish (salmon, mackerel)",
                    "Garlic and herbs (instead of salt)",
                    "Low-fat dairy products"
                ],
                "avoid": [
                    "High-sodium foods (canned soups, processed meats)",
                    "Pickled and fermented foods",
                    "Fast food and restaurant meals",
                    "Alcohol",
                    "Caffeine in excess",
                    "Frozen dinners",
                    "Salty snacks (chips, pretzels)"
                ]
            },
            "thyroid": {
                "recommended": [
                    "Selenium-rich foods (Brazil nuts, tuna)",
                    "Zinc-rich foods (oysters, beef, chickpeas)",
                    "Antioxidant-rich berries",
                    "Bone broth",
                    "Seaweed (for hypothyroidism - consult doctor)",
                    "Lean proteins",
                    "Cruciferous veggies (cooked, not raw)"
                ],
                "avoid": [
                    "Soy-based products (can interfere with medication)",
                    "Excessive iodine supplements",
                    "Processed foods",
                    "Gluten (if sensitive)",
                    "Raw cruciferous vegetables in large amounts",
                    "Sugar and refined carbs",
                    "Alcohol"
                ]
            },
            "heart disease": {
                "recommended": [
                    "Fatty fish (salmon, tuna, mackerel)",
                    "Oats and barley",
                    "Berries and cherries",
                    "Nuts (walnuts, almonds)",
                    "Olive oil",
                    "Avocados",
                    "Dark chocolate (70%+ cocoa)"
                ],
                "avoid": [
                    "Trans fats (fried foods, baked goods)",
                    "Red meat and processed meats",
                    "Full-fat dairy",
                    "Excessive sodium",
                    "Sugar-sweetened beverages",
                    "Refined carbohydrates",
                    "Excessive alcohol"
                ]
            }
        }
        
        # Default plan for unspecified diseases
        default_plan = {
            "recommended": [
                "Fresh fruits and vegetables",
                "Lean proteins (chicken, fish, legumes)",
                "Whole grains",
                "Healthy fats (avocado, nuts, olive oil)",
                "Plenty of water",
                "Herbal teas",
                "Probiotic foods (yogurt, kefir)"
            ],
            "avoid": [
                "Processed foods",
                "Excessive sugar",
                "Fried and fatty foods",
                "Excessive alcohol",
                "Caffeine late in the day",
                "Artificial additives",
                "Highly salted foods"
            ]
        }
        
        # Find matching diet plan
        selected_plan = None
        for key in diet_plans:
            if key in disease:
                selected_plan = diet_plans[key]
                break
        
        if not selected_plan:
            selected_plan = default_plan
        
        # Adjust based on dietary preference
        if request.preference and request.preference != "No Preference":
            pref = request.preference.lower()
            if "vegetarian" in pref:
                selected_plan["recommended"] = [f for f in selected_plan["recommended"] 
                                               if "chicken" not in f.lower() and "fish" not in f.lower()]
                selected_plan["recommended"].append("Plant-based proteins (tofu, tempeh, legumes)")
            elif "vegan" in pref:
                selected_plan["recommended"] = [f for f in selected_plan["recommended"] 
                                               if "chicken" not in f.lower() and "fish" not in f.lower() 
                                               and "yogurt" not in f.lower() and "dairy" not in f.lower()]
                selected_plan["recommended"].extend(["Plant-based proteins", "Fortified plant milks"])
            elif "keto" in pref:
                selected_plan["recommended"] = [f for f in selected_plan["recommended"] 
                                               if "grains" not in f.lower() and "oats" not in f.lower()]
                selected_plan["recommended"].extend(["Healthy fats", "Low-carb vegetables"])
        
        return DiseaseResponse(
            recommended=selected_plan["recommended"][:8],  # Limit to 8 items
            avoid=selected_plan["avoid"][:6]  # Limit to 6 items
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Root endpoint
@app.get("/")
async def root():
    return {
        "message": "Welcome to AroMi AI Agent API",
        "version": "1.0.0",
        "endpoints": [
            "/health - Health check",
            "/generate - Content generation",
            "/fitness-plan - Fitness planning",
            "/disease-diet - Diet planning"
        ]
    }

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)