function [] = plot_constellation_color(corenum,Core_rx_symbol_S_test,Core_rx_symbol_P_test)
figure;
for i1 = 1:corenum
subplot(corenum,2,i1);scatplot(real(Core_rx_symbol_S_test(i1,:)),imag(Core_rx_symbol_S_test(i1,:)));box on;grid on;title({['Core',num2str(i1)]});
end
for i1 = 1:corenum
subplot(corenum,2,i1+corenum);scatplot(real(Core_rx_symbol_P_test(i1,:)),imag(Core_rx_symbol_P_test(i1,:)));box on;grid on;title({['Core',num2str(i1)]});
end
return;
end