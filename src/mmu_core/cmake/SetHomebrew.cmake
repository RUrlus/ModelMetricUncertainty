# Resolve Homebrew install prefix for macOS dependency fallbacks.
if(NOT DEFINED MMU_HOMEBREW_PREFIX)
  if(DEFINED ENV{HOMEBREW_PREFIX} AND IS_DIRECTORY "$ENV{HOMEBREW_PREFIX}")
    set(MMU_HOMEBREW_PREFIX "$ENV{HOMEBREW_PREFIX}")
  elseif(IS_DIRECTORY /opt/homebrew)
    set(MMU_HOMEBREW_PREFIX /opt/homebrew)
  else()
    set(MMU_HOMEBREW_PREFIX /usr/local)
  endif()
endif()

