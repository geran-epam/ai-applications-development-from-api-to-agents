import contextlib
import sys
from pathlib import Path

from fastmcp import FastMCP

from commons.user_service.client import UserServiceClient
from commons.user_service.user_info import UserSearchRequest, UserCreate, UserUpdate

mcp = FastMCP(name="users-management-mcp-server")
user_client = UserServiceClient()


# ==================== TOOLS ====================
@mcp.tool()
async def get_user_by_id(user_id: int) -> str:
    """Provides full user information by user ID."""
    return user_client.get_user(user_id)


@mcp.tool()
async def delete_user(user_id: int) -> str:
    """Deletes a user by user ID. Irreversible."""
    return user_client.delete_user(user_id)


@mcp.tool()
async def search_user(
        name: str | None = None,
        surname: str | None = None,
        email: str | None = None,
        gender: str | None = None,
) -> str:
    """Searches users by name, surname, email and/or gender (partial, case-insensitive match; gender is exact).
    All parameters are optional and combined as filters."""
    request = UserSearchRequest(name=name, surname=surname, email=email, gender=gender)
    # UserServiceClient prints to stdout, which would corrupt the JSON-RPC stream in STDIO transport
    with contextlib.redirect_stdout(sys.stderr):
        return user_client.search_users(**request.model_dump())


@mcp.tool()
async def add_user(user_create_model: UserCreate) -> str:
    """Adds a new user to the system. Required fields: name, surname, email, about_me."""
    return user_client.add_user(user_create_model)


@mcp.tool()
async def update_user(user_id: int, user_update_model: UserUpdate) -> str:
    """Updates an existing user by ID. Provide only the fields that must change."""
    return user_client.update_user(user_id, user_update_model)


# ==================== MCP RESOURCES ====================
@mcp.resource(
    uri="users-management://flow-diagram",
    mime_type="image/png",
    description="Flow diagram (PNG) of the Users Management Agent, MCP client/server and User Service.",
)
async def get_flow_diagram() -> bytes:
    """Returns the flow diagram image as bytes."""
    return (Path(__file__).parent / "flow.png").read_bytes()


# ==================== MCP PROMPTS ====================
@mcp.prompt(description="Guidance for searching through the user database effectively.")
async def user_search_prompt() -> str:
    """Guidance for searching through the user database effectively."""
    return """
You are helping users search through a dynamic user database. The database contains 
realistic synthetic user profiles with the following searchable fields:

## Available Search Parameters
- **name**: First name (partial matching, case-insensitive)
- **surname**: Last name (partial matching, case-insensitive)  
- **email**: Email address (partial matching, case-insensitive)
- **gender**: Exact match (male, female, other, prefer_not_to_say)

## Search Strategy Guidance

### For Name Searches
- Use partial names: "john" finds John, Johnny, Johnson, etc.
- Try common variations: "mike" vs "michael", "liz" vs "elizabeth"
- Consider cultural name variations

### For Email Searches  
- Search by domain: "gmail" for all Gmail users
- Search by name patterns: "john" for emails containing john
- Use company names to find business emails

### For Demographic Analysis
- Combine gender with other criteria for targeted searches
- Use broad searches first, then narrow down

### Effective Search Combinations
- Name + Gender: Find specific demographic segments
- Email domain + Surname: Find business contacts
- Partial names: Cast wider nets for common names

## Example Search Patterns
```
"Find all Johns" → name="john"
"Gmail users named Smith" → email="gmail" + surname="smith"  
"Female users with company emails" → gender="female" + email="company"
"Users with Johnson surname" → surname="johnson"
```

## Tips for Better Results
1. Start broad, then narrow down
2. Try variations of names (John vs Johnny)
3. Use partial matches creatively
4. Combine multiple criteria for precision
5. Remember searches are case-insensitive

When helping users search, suggest multiple search strategies and explain 
why certain approaches might be more effective for their goals.
"""


@mcp.prompt(description="Guidance for creating realistic user profiles.")
async def user_creation_prompt() -> str:
    """Guidance for creating realistic user profiles."""
    return """
You are helping create realistic user profiles for the system. Follow these guidelines 
to ensure data consistency and realism.

## Required Fields
- **name**: 2-50 characters, letters only, culturally appropriate
- **surname**: 2-50 characters, letters only  
- **email**: Valid format, must be unique in system
- **about_me**: Rich, realistic biography (see guidelines below)

## Optional Fields Best Practices
- **phone**: Use E.164 format (+1234567890) when possible
- **date_of_birth**: YYYY-MM-DD format, realistic ages (18-80)
- **gender**: Use standard values (male, female, other, prefer_not_to_say)
- **company**: Real-sounding company names
- **salary**: $30,000-$200,000 range for employed individuals

## Address Guidelines
Provide complete, realistic addresses:
- **country**: Full country names
- **city**: Actual city names  
- **street**: Realistic street addresses
- **flat_house**: Apartment/unit format (Apt 123, Unit 5B, Suite 200)

## Credit Card Guidelines  
Generate realistic but non-functional card data:
- **num**: 16 digits formatted as XXXX-XXXX-XXXX-XXXX
- **cvv**: 3 digits (000-999)
- **exp_date**: MM/YYYY format, future dates only

## Biography Creation ("about_me")
Create engaging, realistic biographies that include:

### Personality Elements
- 1-3 personality traits (curious, adventurous, analytical, etc.)
- Authentic voice and writing style
- Cultural and demographic appropriateness

### Interests & Hobbies  
- 2-4 specific hobbies or activities
- 1-3 broader interests or passion areas
- 1-2 life goals or aspirations

### Biography Templates
Use varied narrative structures:
- "I'm a [trait] person who loves [hobbies]..."
- "When I'm not working, you can find me [activity]..."  
- "Life is all about balance for me. I enjoy [interests]..."
- "As someone who's [trait], I find great joy in [hobby]..."

## Data Validation Reminders
- Email uniqueness is enforced (check existing users)
- Phone numbers should follow consistent formatting
- Date formats must be exact (YYYY-MM-DD)
- Credit card expiration dates must be in the future
- Salary values should be realistic for the demographic

## Cultural Sensitivity
- Match names to appropriate cultural backgrounds
- Consider regional variations in address formats
- Use realistic company names for the user's location
- Ensure hobbies and interests are culturally appropriate

When creating profiles, aim for diversity in:
- Geographic representation
- Age distribution  
- Interest variety
- Socioeconomic backgrounds
- Cultural backgrounds
"""
