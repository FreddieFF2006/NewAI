"""
Diagnostic script to troubleshoot .env file issues
"""

import os
import sys

print("=" * 60)
print("ENV FILE DIAGNOSTIC TOOL")
print("=" * 60)
print()

# Check 1: Current directory
print("1. Current Directory:")
print(f"   {os.getcwd()}")
print()

# Check 2: Does .env file exist?
print("2. Checking for .env file:")
env_path = os.path.join(os.getcwd(), '.env')
if os.path.exists('.env'):
    print("   ✓ .env file EXISTS")
    print(f"   Path: {env_path}")
else:
    print("   ✗ .env file NOT FOUND")
    print(f"   Looking for: {env_path}")
    print()
    print("   FIX: Create .env file in this directory:")
    print(f"   {os.getcwd()}")
    sys.exit(1)
print()

# Check 3: File contents
print("3. .env File Contents:")
try:
    with open('.env', 'r', encoding='utf-8') as f:
        contents = f.read()
    
    print("   --- START OF FILE ---")
    print(contents)
    print("   --- END OF FILE ---")
    print()
    
    # Check for common issues
    lines = contents.strip().split('\n')
    for i, line in enumerate(lines, 1):
        line = line.strip()
        if not line or line.startswith('#'):
            continue
        
        print(f"   Line {i}: {repr(line)}")
        
        if '=' not in line:
            print("   ⚠️  WARNING: No '=' sign found")
        elif ' = ' in line:
            print("   ⚠️  WARNING: Spaces around '=' (should be no spaces)")
        elif line.count('=') > 1:
            print("   ⚠️  WARNING: Multiple '=' signs")
        
        if line.startswith('CLAUDE_API_KEY'):
            parts = line.split('=', 1)
            if len(parts) == 2:
                key = parts[1].strip()
                if key.startswith('"') or key.startswith("'"):
                    print("   ⚠️  WARNING: API key has quotes (remove them)")
                if not key.startswith('sk-ant-'):
                    print("   ⚠️  WARNING: API key doesn't start with 'sk-ant-'")
                else:
                    print(f"   ✓ API key format looks correct (starts with sk-ant-)")

except Exception as e:
    print(f"   ✗ Error reading file: {e}")
    sys.exit(1)
print()

# Check 4: python-dotenv installed?
print("4. Checking python-dotenv:")
try:
    import dotenv
    print("   ✓ python-dotenv is installed")
    print(f"   Version: {dotenv.__version__ if hasattr(dotenv, '__version__') else 'unknown'}")
except ImportError:
    print("   ✗ python-dotenv is NOT installed")
    print()
    print("   FIX: Install it with:")
    print("   pip install python-dotenv")
    sys.exit(1)
print()

# Check 5: Load .env and check environment
print("5. Loading .env file:")
try:
    from dotenv import load_dotenv
    result = load_dotenv()
    print(f"   load_dotenv() returned: {result}")
    
    api_key = os.getenv('CLAUDE_API_KEY')
    if api_key:
        masked = api_key[:10] + "..." + api_key[-4:] if len(api_key) > 14 else "***"
        print(f"   ✓ CLAUDE_API_KEY loaded: {masked}")
        print()
        print("=" * 60)
        print("✅ SUCCESS! Everything looks good!")
        print("=" * 60)
        print()
        print("Your .env file is working correctly.")
        print("You should now be able to run: python example_usage.py")
    else:
        print("   ✗ CLAUDE_API_KEY not found in environment")
        print()
        print("PROBLEM: The .env file exists but the key isn't being loaded.")
        print()
        print("POSSIBLE CAUSES:")
        print("1. Wrong format in .env file")
        print("2. File encoding issue")
        print("3. Cached Python bytecode")
        print()
        print("TRY THIS:")
        print("1. Delete .env file")
        print("2. Create new one with EXACT format:")
        print("   CLAUDE_API_KEY=sk-ant-your-key-here")
        print("   (no spaces, no quotes, no extra lines)")
        
except Exception as e:
    print(f"   ✗ Error loading .env: {e}")
    sys.exit(1)
print()

# Check 6: Show correct format
if not os.getenv('CLAUDE_API_KEY'):
    print("=" * 60)
    print("CORRECT .env FILE FORMAT:")
    print("=" * 60)
    print()
    print("Your .env file should contain EXACTLY this:")
    print()
    print("CLAUDE_API_KEY=sk-ant-your-actual-key-here")
    print()
    print("Rules:")
    print("  ✓ No spaces around =")
    print("  ✓ No quotes around the key")
    print("  ✓ No 'export' command")
    print("  ✓ Key starts with sk-ant-")
    print()
