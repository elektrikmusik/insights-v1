"""
Supabase JWT authentication middleware.
"""
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer
from supabase import create_client

from insights.core.config import settings

security = HTTPBearer()

async def verify_jwt(request: Request):
    """
    Validate Supabase JWT and attach user to request state.
    """
    auth_header = request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        # Skip auth for health checks if needed, but usually handled by router-level dependencies
        return
    
    token = auth_header.split(" ")[1]
    
    try:
        # Create a client for verification
        # Note: In production, consider using a faster verification method (e.g. jose) 
        # to avoid a roundtrip to Supabase for every request if possible.
        supabase = create_client(settings.SUPABASE_URL, settings.SUPABASE_ANON_KEY)
        user_response = supabase.auth.get_user(token)
        
        if not user_response.user:
            raise HTTPException(status_code=401, detail="Invalid token")
            
        request.state.user = user_response.user
        request.state.user_id = user_response.user.id
        
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"Authentication failed: {str(e)}")
