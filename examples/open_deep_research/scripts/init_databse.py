#!/usr/bin/env python3
"""
API Database Initialization Script
=================================

This script initializes all API databases used by the agent system.
It imports and runs the build functions from individual API database builders.

Currently supported APIs:
- MediaWiki Action API (Read-only, no authentication required)
- YouTube Data API v3 (API Key required for most endpoints)
- ORCID Public API (Read-only, token required for some endpoints)

Usage:
    python scripts/init_databse.py [options]

Options:
    --all           Build all API databases (default)
    --mediawiki     Build only MediaWiki API database
    --youtube       Build only YouTube API database
    --orcid         Build only ORCID API database
    --clean         Clean existing generated files before building
    --quiet         Suppress detailed output

Output:
    For each API, the following files are generated in scripts/init_api_database/:
    - {api_name}_api.json        : Structured JSON database for programmatic access
    - {api_name}_api_llm.md      : LLM-friendly Markdown documentation
    - {api_name}_api_summary.md  : Quick reference endpoint summary
"""

import argparse
import os
import sys
import importlib.util
from pathlib import Path
from typing import List, Callable


def get_script_dir() -> Path:
    """Get the directory containing this script."""
    return Path(__file__).parent


def get_api_db_dir() -> Path:
    """Get the API database directory."""
    return get_script_dir() / "init_api_database"


def load_builder_module(module_name: str):
    """
    Dynamically load a builder module from init_api_database directory.
    
    Args:
        module_name: Name of the module (without .py extension)
        
    Returns:
        Loaded module object
    """
    module_path = get_api_db_dir() / f"{module_name}.py"
    if not module_path.exists():
        raise FileNotFoundError(f"Builder module not found: {module_path}")
    
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    
    # Add the init_api_database directory to sys.path temporarily
    # so that imports within the module work correctly
    api_db_dir = str(get_api_db_dir())
    if api_db_dir not in sys.path:
        sys.path.insert(0, api_db_dir)
    
    try:
        spec.loader.exec_module(module)
    finally:
        # Clean up sys.path
        if api_db_dir in sys.path:
            sys.path.remove(api_db_dir)
    
    return module


def clean_generated_files(quiet: bool = False):
    """
    Remove all generated API database files.
    
    Args:
        quiet: If True, suppress output messages
    """
    api_db_dir = get_api_db_dir()
    patterns = ["*.json", "*_llm.md", "*_summary.md"]
    
    removed_count = 0
    for pattern in patterns:
        for file_path in api_db_dir.glob(pattern):
            # Don't delete source .py files
            if file_path.suffix == ".py":
                continue
            try:
                file_path.unlink()
                removed_count += 1
                if not quiet:
                    print(f"  Removed: {file_path.name}")
            except Exception as e:
                print(f"  Warning: Could not remove {file_path.name}: {e}")
    
    if not quiet:
        print(f"Cleaned {removed_count} generated files.\n")


def build_mediawiki_api(quiet: bool = False):
    """Build MediaWiki API database."""
    if not quiet:
        print("=" * 60)
        print("Building MediaWiki Action API Database")
        print("=" * 60)
    
    try:
        module = load_builder_module("build_mediawiki_api_db")
        module.main()
        if not quiet:
            print()
    except Exception as e:
        print(f"Error building MediaWiki API: {e}")
        raise


def build_youtube_api(quiet: bool = False):
    """Build YouTube Data API database."""
    if not quiet:
        print("=" * 60)
        print("Building YouTube Data API v3 Database")
        print("=" * 60)
    
    try:
        module = load_builder_module("build_youtube_api_db")
        module.main()
        if not quiet:
            print()
    except Exception as e:
        print(f"Error building YouTube API: {e}")
        raise


def build_orcid_api(quiet: bool = False):
    """Build ORCID Public API database."""
    if not quiet:
        print("=" * 60)
        print("Building ORCID Public API Database")
        print("=" * 60)
    
    try:
        module = load_builder_module("build_orcid_api_db")
        module.main()
        if not quiet:
            print()
    except Exception as e:
        print(f"Error building ORCID API: {e}")
        raise


def build_all(quiet: bool = False):
    """Build all API databases."""
    builders = [
        ("MediaWiki", build_mediawiki_api),
        ("YouTube", build_youtube_api),
        ("ORCID", build_orcid_api),
    ]
    
    success_count = 0
    failed = []
    
    for name, builder in builders:
        try:
            builder(quiet)
            success_count += 1
        except Exception as e:
            failed.append((name, str(e)))
            print(f"Failed to build {name} API: {e}\n")
    
    return success_count, failed


def print_summary(success_count: int, failed: List[tuple], quiet: bool = False):
    """Print build summary."""
    if quiet:
        return
    
    print("=" * 60)
    print("BUILD SUMMARY")
    print("=" * 60)
    
    api_db_dir = get_api_db_dir()
    
    # List generated files
    json_files = list(api_db_dir.glob("*.json"))
    md_files = list(api_db_dir.glob("*.md"))
    
    print(f"\n✅ Successfully built: {success_count} API database(s)")
    
    if failed:
        print(f"\n❌ Failed: {len(failed)} API database(s)")
        for name, error in failed:
            print(f"   - {name}: {error}")
    
    print(f"\n📁 Generated files in: {api_db_dir}")
    print(f"   - JSON databases: {len(json_files)}")
    for f in sorted(json_files):
        size_kb = f.stat().st_size / 1024
        print(f"     • {f.name} ({size_kb:.1f} KB)")
    
    print(f"   - Markdown docs: {len(md_files)}")
    for f in sorted(md_files):
        size_kb = f.stat().st_size / 1024
        print(f"     • {f.name} ({size_kb:.1f} KB)")
    
    print("\n" + "=" * 60)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Initialize API databases for the agent system",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/init_databse.py              # Build all databases
  python scripts/init_databse.py --mediawiki  # Build only MediaWiki
  python scripts/init_databse.py --clean      # Clean and rebuild all
  python scripts/init_databse.py --quiet      # Build with minimal output
        """
    )
    
    parser.add_argument(
        "--all", 
        action="store_true", 
        default=True,
        help="Build all API databases (default behavior)"
    )
    parser.add_argument(
        "--mediawiki", 
        action="store_true",
        help="Build only MediaWiki API database"
    )
    parser.add_argument(
        "--youtube", 
        action="store_true",
        help="Build only YouTube API database"
    )
    parser.add_argument(
        "--orcid", 
        action="store_true",
        help="Build only ORCID API database"
    )
    parser.add_argument(
        "--clean", 
        action="store_true",
        help="Clean existing generated files before building"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress detailed output"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # Check if specific API is requested
    specific_api = args.mediawiki or args.youtube or args.orcid
    
    if not args.quiet:
        print("\n" + "=" * 60)
        print("  API DATABASE INITIALIZATION")
        print("=" * 60 + "\n")
    
    # Clean if requested
    if args.clean:
        if not args.quiet:
            print("Cleaning existing generated files...")
        clean_generated_files(args.quiet)
    
    # Build requested databases
    success_count = 0
    failed = []
    
    if specific_api:
        # Build only specified APIs
        if args.mediawiki:
            try:
                build_mediawiki_api(args.quiet)
                success_count += 1
            except Exception as e:
                failed.append(("MediaWiki", str(e)))
        
        if args.youtube:
            try:
                build_youtube_api(args.quiet)
                success_count += 1
            except Exception as e:
                failed.append(("YouTube", str(e)))
        
        if args.orcid:
            try:
                build_orcid_api(args.quiet)
                success_count += 1
            except Exception as e:
                failed.append(("ORCID", str(e)))
    else:
        # Build all APIs
        success_count, failed = build_all(args.quiet)
    
    # Print summary
    print_summary(success_count, failed, args.quiet)
    
    # Return exit code based on success
    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
