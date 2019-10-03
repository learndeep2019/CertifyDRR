function genericcallback (t, f, x, clr)

  if nargin < 4
    fprintf('%3d  %0.5g \n', t, f);
  else
    cprintf(clr, '%3d  %0.5g\n', t, f);
  end
end