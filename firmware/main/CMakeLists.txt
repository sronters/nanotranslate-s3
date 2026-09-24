idf_component_register(
  SRCS "main.cpp" "ui.cpp" "../../runtime/src/nanomt.cpp"
  INCLUDE_DIRS "." "../../runtime/include"
  REQUIRES esp_lcd driver esp_timer
)
